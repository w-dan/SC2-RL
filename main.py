import multiprocessing
import platform

import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from sc2_rl.rl.deepq import RandomPolicy
from sc2_rl.rl.sc2env import create_environment

MAP_NAME = "Scorpion_1.01"
verbose = 2


def preprocess_observation(observation, target_shape=(224, 224)):
    # Resize observation to the target shape
    observation = tf.image.resize(observation, target_shape)
    return observation


def compute_avg_return(env, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0

        print(f"time step: {time_step}")
        print(f"time step type: {type(time_step)}")

        while not time_step.is_last():
            # Preprocess observation to match model's expected input shape
            time_step = ts.TimeStep(
                step_type=time_step.step_type,
                reward=time_step.reward,
                discount=time_step.discount,
                observation=preprocess_observation(time_step.observation),
            )
            action_step = policy.take_action(time_step)
            time_step = env.step(action_step)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


def main():
    if platform.system() == "Windows":
        print("[+] Freezing multiprocessing support...")
        multiprocessing.freeze_support()

    env = create_environment(MAP_NAME, verbose=verbose)

    policy = RandomPolicy(env.time_step_spec(), env.action_spec())

    avg_return = compute_avg_return(env, policy, 10)
    print(f"Average return: {avg_return}")


if __name__ == "__main__":
    main()
