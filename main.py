import multiprocessing
import platform

from sc2 import maps  # maps method for loading maps to play in.
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import (  # function that facilitates actually running the agents in games
    run_game,
)
from sc2.player import (  # wrapper for whether or not the agent is one of your bots, or a "computer" player
    Bot,
    Computer,
)
from tf_agents.environments import TFPyEnvironment

from sc2_rl.bots.artanis_bot import ArtanisBot
from sc2_rl.rl.deepq import RandomPolicy
from sc2_rl.rl.sc2env import create_environment

MAP_NAME = "Scorpion_1.01"
verbose = 1

# https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/suite_gym/wrap_env


# == PROBLEM ==
# = Seems like GymEnv needs more functions
def compute_avg_return(
    environment: TFPyEnvironment, policy: RandomPolicy, num_episodes=10
):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()  # tuple (observation, reward, done, info)

        observation = time_step[0]
        reward = time_step[1]
        done = time_step[2]
        info = time_step[3]

        episode_return = 0.0

        print(f"time step: {time_step}")
        print(f"time step type: {type(time_step)}")

        while not time_step.is_last():
            action_step = policy.take_action(time_step)
            time_step = environment.step(action_step)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes

    return avg_return.numpy()[0]


if __name__ == "__main__":

    operating_system = platform.system()

    if operating_system == "Windows":
        print("[+] Freezing multiprocessing support...")
        multiprocessing.freeze_support()  # Only needed on Windows

    env = create_environment(ArtanisBot(verbose), MAP_NAME, verbose=verbose)
    policy = RandomPolicy(env.time_step_spec(), env.action_spec())

    compute_avg_return(env, policy, 1)

    # MAP_NAME = "Scorpion_1.01"

    # run_game(  # run_game is a function that runs the game.
    #     maps.get(MAP_NAME), # the map we are playing on
    #     [Bot(Race.Protoss, ArtanisBot(1)), # runs our coded bot, protoss race, and we pass our bot object
    #     Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
    #     realtime=False, # When set to True, the agent is limited in how long each step can take to process.
    # )
