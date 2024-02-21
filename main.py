import multiprocessing
import os
import platform

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from sc2_rl.rl.dqn import RandomPolicy
from sc2_rl.rl.sc2env import create_environment

MAP_NAME = "Scorpion_1.01"
verbose = 2


def compute_avg_return(env, agent, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward.numpy()
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return


def main():
    if platform.system() == "Windows":
        print("[+] Freezing multiprocessing support...")
        multiprocessing.freeze_support()

    env = create_environment(MAP_NAME, verbose=verbose)

    train_env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(env)

    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=(100, 50, 25),
        preprocessing_layers=tf.keras.layers.Lambda(lambda x: x / 255.0),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    global_step = tf.Variable(0)

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=10000,
        # device="gpu:*",
    )
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=5000,
    )

    # Initial data collection
    collect_driver.run()

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2,
        single_deterministic_pass=False,
    ).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    def train_one_iteration():
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        iteration = agent.train_step_counter.numpy()
        print("iteration: {0} loss: {1}".format(iteration, train_loss.loss))

    os.makedirs("output", exist_ok=True)

    checkpoint_dir = os.path.join("output", "checkpoint")
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step,
    )

    policy_dir = os.path.join("output", "policy")
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    print("Training one iteration....")
    train_one_iteration()

    train_checkpointer.save(global_step)
    tf_policy_saver.save(policy_dir)

    avg_return = compute_avg_return(eval_env, agent, 1)
    print(f"Average return: {avg_return}")


if __name__ == "__main__":
    main()
