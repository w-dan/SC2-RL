import json
import platform
import subprocess
import time
from enum import Enum

import cv2
import numpy as np
import redis
import tensorflow as tf
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import sc2_rl.types.game as game_info
import sc2_rl.types.rewards as rwd


# https://gymnasium.farama.org/api/env/
class Sc2Env(py_environment.PyEnvironment):
    def __init__(self, map_name: str, verbose: int = 3):
        super(Sc2Env, self).__init__()

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.redis_client.flushall()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(224, 224, 3),
            dtype=np.float32,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)
        self._episode_ended = False
        self.verbose = verbose
        self.map_name = map_name
        self.game_status = game_info.GameResult.PLAYING
        self.game_tick = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
        self.acmrwd = 0.0
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)
        self._episode_ended = False
        self.game_status = game_info.GameResult.PLAYING

        if hasattr(self, "game_process") and self.game_process.poll() is None:
            self.game_process.kill()
            self.game_process.wait()

        if platform.system() == "Windows":
            self.game_process = subprocess.Popen(
                [
                    ".venv/Scripts/python.exe",
                    "sc2_rl/bots/artanis_bot.py",
                ],
            )
        elif platform.system() == "Linux":
            self.game_process = subprocess.Popen(
                [
                    "python",
                    "sc2_rl/sc2/artanis_bot.py",
                ],
            )

        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # If the episode ended, automatically reset the environment
            return self._reset()

        try:
            self.redis_client.rpush("action_queue", int(action))
            _, state_rwd_action = self.redis_client.blpop("state_queue", timeout=0)
        except Exception as e:
            # If redis is not up, then kill game if executed
            print(f"There was an error using redis!!\n{e}")
            print(f"Shutting down game...")
            if hasattr(self, "game_process") and self.game_process.poll() is None:
                self.game_process.kill()
                self.game_process.wait()
            print("Game is down!")
            exit(2)

        state_rwd_action = json.loads(state_rwd_action.decode())

        self._state = np.array(state_rwd_action["state"], dtype=np.uint8)
        micro_reward = state_rwd_action["micro-reward"]
        game_tick = state_rwd_action["info"]["game_tick"]
        self.game_status = state_rwd_action["game_status"]

        self.game_tick = game_tick if game_tick is not None else self.game_tick + 1

        if self.game_status == game_info.GameResult.PLAYING:
            reward = micro_reward
            self._episode_ended = False
        else:
            reward = self._calculate_macro_reward(state_rwd_action["info"])
            self._episode_ended = True

        self.acmrwd += reward

        if self.verbose >= 2:
            cv2.imshow(
                "map",
                cv2.flip(
                    cv2.resize(
                        self._state, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
                    ),
                    0,
                ),
            )
            cv2.waitKey(1)

        if self.verbose >= 3:
            # save map image into "replays dir"
            cv2.imwrite(f"replays/{int(time.time())}-{self.game_tick}.png", self._state)

        if self.verbose >= 1:
            info = state_rwd_action["info"]
            if (
                self.game_tick is not None and self.game_tick % 100 == 0
            ) or self._episode_ended:
                print(
                    f"Game Tick: {self.game_tick}. Total reward: {self.acmrwd:.4f}. Void Ray: {info['n_VOIDRAY']}"
                )

        return (
            ts.termination(self._state, reward)
            if self._episode_ended
            else ts.transition(self._state, reward, discount=1.0)
        )

    def _calculate_macro_reward(self, info) -> float:
        reward = 0

        if self.game_status == game_info.GameResult.VICTORY:
            reward += rwd.GAME_REWARD.WIN
        elif self.game_status == game_info.GameResult.DEFEAT:
            reward += rwd.GAME_REWARD.LOSE

        return reward


def preprocess_observation(observation, target_shape=(224, 224)):
    # Resize observation to the target shape
    observation = tf.convert_to_tensor(observation)
    observation = tf.cast(observation, tf.float32)
    observation = tf.image.resize(observation, target_shape)
    return observation


class PreprocessEnvironmentWrapper(py_environment.PyEnvironment):
    def __init__(self, env):
        super().__init__()
        self._env = env

    def preprocess_observation(self, observation):
        return preprocess_observation(observation)

    def _step(self, action):
        time_step = self._env.step(action)
        processed_observation = self.preprocess_observation(time_step.observation)
        return ts.TimeStep(
            time_step.step_type,
            time_step.reward,
            time_step.discount,
            processed_observation,
        )

    def _reset(self):
        time_step = self._env.reset()
        processed_observation = self.preprocess_observation(time_step.observation)
        return ts.TimeStep(
            time_step.step_type,
            time_step.reward,
            time_step.discount,
            processed_observation,
        )

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def get_info(self):
        return self._env.get_info()

    def get_state(self):
        return self._env.get_state()

    def set_state(self, state):
        return self._env.set_state(state)


def create_environment(map_name: str, verbose: int):
    return PreprocessEnvironmentWrapper(Sc2Env(map_name, verbose))
