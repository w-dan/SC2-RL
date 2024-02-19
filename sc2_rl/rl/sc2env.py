from enum import Enum
import cv2
import time
import gymnasium
import numpy as np

import gymnasium as spaces
from gymnasium.spaces import Discrete, Box

from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.

from multiprocessing import Process, Value


class GameResult(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2

# https://gymnasium.farama.org/api/env/
class Sc2Env(gymnasium.Env):
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
        # https://gymnasium.farama.org/api/spaces/#spaces
        self.action_space = Discrete(3)
        # I think this is what you can render later
        self.observation_space = Box(low=0, high=255, shape=(224, 224,3), dtype=np.uint8)
        self.bot = bot
        self.result = Value('d', GameResult.PLAYING.value)
        self.game = Process(target=self._run_game)
        self.verbose = verbose
        self.map_name = map_name

    def step(self, action):
        state_rwd = self.bot.step(action)

        observation = state_rwd["state"]["map"]
        self.observation_space = observation
        reward = state_rwd["micro-reward"]
        reward += self._calculate_macro_reward(state_rwd["info"])
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

        done = False if self.result.value == GameResult.PLAYING.value else True

        return observation, reward, done, info

    def reset(self):
        print("=== ENVIRONMENT RESET ===")
        self.game.start()

        # state_rwd = self.bot.get_state_from_world()
        # observation = state_rwd["state"]["map"]
        observation = np.zeros((244, 244, 3), dtype=np.uint8)
        reward = 0
        info = None
        
        return observation, reward, False, info

    def _run_game(self):
        result = run_game(  # run_game is a function that runs the game.
            maps.get(self.map_name), # the map we are playing on
            [Bot(Race.Protoss, self.bot), # runs our coded bot, protoss race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )

        if self.verbose >= 1:
            print(f"== RESULT : {str(result)} ==")

        self.result.value = str(result)

    def current_timestep(self) -> float:
        # TODO : RETURN VALUE OF TIME ON GAME
        return 0

    def _calculate_macro_reward(self, info) -> float:
        reward = 0
        
        if self.result.value == GameResult.WIN.value:
            reward += 500
        elif self.result.value == GameResult.LOSE.value:
            reward += -500
        
        return reward

class PreprocessWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(224, 224, 3), dtype=np.float32)

    def observation(self, obs):
        # Assuming obs is an image represented as uint8 values in range [0, 255]
        # Convert the image to float32 and scale to range [0, 1]
        processed_obs = obs.astype(np.float32) / 255.0
        return processed_obs
