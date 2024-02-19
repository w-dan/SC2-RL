from tf_agents.environments.suite_gym import wrap_env

from sc2_rl.bots.artanis_bot import ArtanisBot
from sc2_rl.rl.deepq import RandomPolicy, action_space #, my_random_py_policy
from sc2_rl.rl.sc2env import Sc2Env

MAP_NAME = "Scorpion_1.01"

verbose = 1
env = Sc2Env(ArtanisBot(verbose), MAP_NAME, verbose=verbose)


# run_game(  # run_game is a function that runs the game.
#     maps.get(MAP_NAME), # the map we are playing on
#     [Bot(Race.Protoss, IncrediBot()), # runs our coded bot, protoss race, and we pass our bot object
#      Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
#     realtime=False, # When set to True, the agent is limited in how long each step can take to process.
# )

policy = RandomPolicy(wrap_env(env))


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


compute_avg_return(policy.environment, policy.policy, 1)
