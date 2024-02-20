import json
import subprocess
import time
from enum import Enum

import cv2
import numpy as np
import redis
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from sc2_rl.types.constants import REWARD


class GameResult(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2


# https://gymnasium.farama.org/api/env/
class Sc2Env(py_environment.PyEnvironment):
    def __init__(self, map_name: str, verbose: int = 3):
        super(Sc2Env, self).__init__()

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=2, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(224, 224, 3),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)
        self._episode_ended = False
        self.verbose = verbose
        self.map_name = map_name
        self.result = GameResult.PLAYING
        self.iteration = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # self.future = self.executor.submit(self._start_game)
        print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
        self._state = np.zeros((224, 224, 3), dtype=np.uint8)
        self._episode_ended = False
        self.result = GameResult.PLAYING

        subprocess.Popen(
            [
                "python3",
                f"sc2_rl/bots/artanis_bot.py",
            ],
        )
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # If the episode ended, automatically reset the environment
            return self._reset()

        self.redis_client.rpush("action_queue", int(action))
        _, state_rwd_action = self.redis_client.blpop("state_queue", timeout=0)

        state_rwd_action = json.loads(state_rwd_action.decode())

        self._state = np.array(state_rwd_action["state"], dtype=np.uint8)
        micro_reward = state_rwd_action["micro-reward"]
        iteration = state_rwd_action["info"]["iteration"]
        self.result = state_rwd_action["done"]

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
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", self._state)

        if self.verbose >= 1:
            info = state_rwd_action["info"]
            if iteration % 100 == 0:
                print(
                    f"Iter: {iteration}. RWD: {micro_reward}. Void Ray: {info['n_VOIDRAY']}"
                )

        self.iteration = iteration

        if self.result == GameResult.PLAYING.value:
            reward = micro_reward
            self._episode_ended = False
        else:
            reward = self._calculate_macro_reward(state_rwd_action["info"])
            self._episode_ended = True

        print(self._state.shape)
        return (
            ts.termination(self._state, reward)
            if self._episode_ended
            else ts.transition(self._state, reward, discount=1.0)
        )

    def _calculate_macro_reward(self, info) -> float:
        reward = 0

        if self.result.value == GameResult.WIN.value:
            reward += REWARD.WIN.value
        elif self.result.value == GameResult.LOSE.value:
            reward += REWARD.LOSE.value

        return reward


def create_environment(map_name: str, verbose: int):
    return tf_py_environment.TFPyEnvironment(Sc2Env(map_name, verbose))
