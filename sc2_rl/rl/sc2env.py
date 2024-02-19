from enum import Enum
import cv2
import time
import gymnasium 
import numpy as np

import gymnasium as spaces
from gymnasium.spaces import Discrete, Box
from tf_agents.specs import array_spec, tensor_spec

from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.

from multiprocessing import Process, Value
from tf_agents.environments import gym_wrapper, tf_py_environment
from tf_agents.environments import py_environment
from sc2_rl.types.constants import REWARD
import numpy as np
from enum import Enum
from multiprocessing import Process, Value
import cv2
import time
# from pysc2.env import sc2_env
# from pysc2.lib import actions, features, units
from absl import app
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.trajectories import TimeStep


class GameResult(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2

# https://gymnasium.farama.org/api/env/
class Sc2Env(py_environment.PyEnvironment):
    """
        - Micro reward is managed by Bot
        - Macro reward is managed by Environment
    """
    def __init__(self, bot, map_name: str, verbose: int = 3):
        '''
        - verbose [0, 1, 2, 3]
            (accumulative)
            - 0 : Nothing
            - 1 : prints
            - 2 : Image show
            - 3 : Image save
        '''
        super(Sc2Env, self).__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(224, 224, 3), dtype=np.uint8, minimum=0, maximum=255, name='observation')
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)
        self._episode_ended = False
        self.bot = bot
        self.verbose = verbose
        self.map_name = map_name
        self.result = GameResult.PLAYING
        self.game = Process(target=self._run_game)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("=== ENVIRONMENT RESET ===")
        self.game.start()
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)  # Reset the state
        self._episode_ended = False
        self.result = GameResult.PLAYING
        # Aquí deberías también reiniciar tu bot y el juego de StarCraft II según sea necesario
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Implementa la lógica de tu paso aquí usando `action`
        # Deberías actualizar `self._state` con el nuevo estado
        # y decidir si el episodio ha terminado (`self._episode_ended`)
        # basado en la lógica de tu juego y bot.
        
        state_rwd = self.bot.step(action)

        observation = state_rwd["state"]["map"]
        self.observation_space = observation
        info = state_rwd["info"]
        iteration = info["iteration"]

        if self.verbose >= 2:
            cv2.imshow('map',cv2.flip(cv2.resize(observation, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
            cv2.waitKey(1)

        if self.verbose >= 3:
            # save map image into "replays dir"
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", observation)

        if self.verbose >= 1:
            if iteration % 100 == 0:
                print(f"Iter: {iteration}. RWD: {reward}. Void Ray: {info['n_VOIDRAY']}")

        if self.result == GameResult.PLAYING: #TODO Si add macro sacar de aqui
            reward = state_rwd["micro-reward"]
            self._episode_ended = False
        else:
            reward = self._calculate_macro_reward(state_rwd["info"])
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)

    def _run_game(self):
        result = run_game(  # run_game is a function that runs the game.
            maps.get(self.map_name), # the map we are playing on
            [Bot(Race.Protoss, self.bot), # runs our coded bot, protoss race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )

        while self.bot.state.game_loop < 0:
            continue

        if self.verbose >= 1:
            print(f"== RESULT : {str(result)} ==")

        self.result.value = str(result)

    def _calculate_macro_reward(self, info) -> float:
        reward = 0
        
        if self.result.value == GameResult.WIN.value:
            reward += REWARD.WIN.value
        elif self.result.value == GameResult.LOSE.value:
            reward += REWARD.LOSE.value
        
        return reward

        


# class Sc2Env(gymnasium.Env):
#     """
#         - Micro reward is managed by Bot
#         - Macro reward is managed by Environment
#     """
#     def __init__(self, bot, map_name: str, verbose: int = 3):
#         '''
#         - verbose [0, 1, 2, 3]
#             (accumulative)
#             - 0 : Nothing
#             - 1 : prints
#             - 2 : Image show
#             - 3 : Image save
#         '''
#         super(Sc2Env, self).__init__()
#         # https://gymnasium.farama.org/api/spaces/#spaces
#         self.action_space = Discrete(3)
#         # I think this is what you can render later
#         self.observation_space = Box(low=0, high=255, shape=(224, 224,3), dtype=np.uint8)
#         self.bot = bot
#         self.result = Value('d', GameResult.PLAYING.value)
#         self.game = Process(target=self._run_game)
#         self.verbose = verbose
#         self.map_name = map_name

#     def step(self, action):
#         state_rwd = self.bot.step(action)

#         observation = state_rwd["state"]["map"]
#         self.observation_space = observation
#         reward = state_rwd["micro-reward"]
#         reward += self._calculate_macro_reward(state_rwd["info"])
#         info = state_rwd["info"]
#         iteration = info["iteration"]

#         if self.verbose >= 2:
#             cv2.imshow('map',cv2.flip(cv2.resize(observation, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
#             cv2.waitKey(1)

#         if self.verbose >= 3:
#             # save map image into "replays dir"
#             cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", observation)

#         if self.verbose >= 1:
#             if iteration % 100 == 0:
#                 print(f"Iter: {iteration}. RWD: {reward}. Void Ray: {info['n_VOIDRAY']}")

#         done = False if self.result.value == GameResult.PLAYING.value else True

#         return observation, reward, done, info

#     def reset(self):
#         print("=== ENVIRONMENT RESET ===")
#         self.game.start()

#         # state_rwd = self.bot.get_state_from_world()
#         # observation = state_rwd["state"]["map"]
#         observation = np.zeros((244, 244, 3), dtype=np.uint8)
#         reward = 0
#         info = None
#         done = False

#         return observation, reward, done, info

#     def _run_game(self):
#         result = run_game(  # run_game is a function that runs the game.
#             maps.get(self.map_name), # the map we are playing on
#             [Bot(Race.Protoss, self.bot), # runs our coded bot, protoss race, and we pass our bot object 
#             Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
#             realtime=False, # When set to True, the agent is limited in how long each step can take to process.
#         )

#         if self.verbose >= 1:
#             print(f"== RESULT : {str(result)} ==")

#         self.result.value = str(result)
    
#     def current_timestep(self) -> float:
#         # TODO : RETURN VALUE OF TIME ON GAME
#         return 0

#     def _calculate_macro_reward(self, info) -> float:
#         reward = 0
        
#         if self.result.value == GameResult.WIN.value:
#             reward += 500
#         elif self.result.value == GameResult.LOSE.value:
#             reward += -500
        
#         return reward

# class PreprocessWrapper(gymnasium.ObservationWrapper):
#     def __init__(self, env):
#         super(PreprocessWrapper, self).__init__(env)
#         self.observation_space = Box(low=0, high=1, shape=(224, 224, 3), dtype=np.float32)

#     def observation(self, obs):
#         # Assuming obs is an image represented as uint8 values in range [0, 255]
#         # Convert the image to float32 and scale to range [0, 1]
#         processed_obs = obs.astype(np.float32) / 255.0
#         return processed_obs

# class CustomGymWrapper(gym_wrapper.GymWrapper):
#     def __init__(self, gym_env, dtype=np.float32):
#         super(CustomGymWrapper, self).__init__(gym_env)
#         self._dtype = dtype
#         self._observation_spec = array_spec.BoundedArraySpec(
#             shape=(224, 224, 3), dtype=dtype, minimum=0, maximum=1, name='observation')

#     def _observation_spec(self):
#         return self._observation_spec

#     def _to_tensor_spec(self, spec):
#         return tensor_spec.from_spec(spec)

#     def _observation(self, observation):
#         return np.array(observation, dtype=self._dtype)

def create_environment(bot, map_name: str, verbose: int):
    return tf_py_environment.TFPyEnvironment(Sc2Env(bot, map_name, verbose))
