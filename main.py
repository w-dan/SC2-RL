import os

import tensorflow as tf
import tqdm
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import TFPolicy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from sc2_rl.rl.dqn import RandomPolicy
from sc2_rl.rl.sc2env import create_environment

MAP_NAME = "Scorpion_1.01"
verbose = 3


def compute_avg_return(env, agent_policy, num_episodes=10):
    total_return = 0.0
    for _ in tqdm.tqdm(range(num_episodes), desc="Agent eval", unit=" games"):
        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = agent_policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward.numpy()
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return


def main(output):
    N = 1000
    num_episodes_per_iteration = 10
    num_iterations = N // num_episodes_per_iteration

    train_env = tf_py_environment.TFPyEnvironment(
        create_environment(MAP_NAME, os.path.abspath("train"), verbose=verbose)
    )
    # eval_env = tf_py_environment.TFPyEnvironment(
    #     create_environment(MAP_NAME, os.path.abspath("env"), verbose=0)
    # )

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers={
            "structures_state": tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        16, (3, 3), activation="relu", input_shape=(224, 224, 3)
                    ),
                    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                    tf.keras.layers.Flatten(),
                ]
            ),
            "units_state": tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        16, (3, 3), activation="relu", input_shape=(224, 224, 3)
                    ),
                    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                    tf.keras.layers.Flatten(),
                ]
            ),
            "minerals": tf.keras.layers.Dense(1),
            "vespene": tf.keras.layers.Dense(1),
            "supply_used": tf.keras.layers.Dense(1),
            "supply_cap": tf.keras.layers.Dense(1),
        },
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        fc_layer_params=(64, 32),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    global_step = global_step = tf.Variable(0, dtype=tf.int64)

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
        gamma=0.99,
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=10,
    )
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=num_episodes_per_iteration,
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        sample_batch_size=8,
        num_steps=2,
        single_deterministic_pass=False,
    ).prefetch(tf.data.experimental.AUTOTUNE)

    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    def train(num_iterations):
        for iteration in range(num_iterations):
            collect_driver.run()

            total_loss = 0
            for _ in range(num_episodes_per_iteration):
                experience, _ = next(iterator)
                train_loss = agent.train(experience)
                total_loss += train_loss.loss

            print(
                f"Iteration: {iteration}, Loss: {total_loss / num_episodes_per_iteration}"
            )

            # if iteration % 10 == 0:
            #     avg_return = compute_avg_return(
            #         eval_env, agent.policy, 10
            #     )  # Evaluar con 10 episodios
            #     print(f"Iteration: {iteration}, Average Return: {avg_return}")

            train_checkpointer.save(global_step)
            tf_policy_saver.save(policy_dir)

    checkpoint_dir = os.path.join("output", "checkpoint")
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step,
    )
    train_checkpointer.initialize_or_restore()
    status = train_checkpointer.initialize_or_restore()
    status.expect_partial()

    policy_dir = os.path.join("output", "policy")
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    print(f"Training {num_iterations} iterations....")
    train(num_iterations)


if __name__ == "__main__":
    main("output/")
    # saved_policy = tf.compat.v2.saved_model.load("output/policy/")

    # env = create_environment(MAP_NAME, verbose=verbose)
    # eval_env = tf_py_environment.TFPyEnvironment(env)

    # print(compute_avg_return(eval_env, saved_policy, 2))
