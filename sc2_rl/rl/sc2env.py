import json
import os
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np
import redis
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import sc2_rl.types.game as game_info
import sc2_rl.types.rewards as rwd
from sc2_rl.types.actions import Action


class Sc2Env(py_environment.PyEnvironment):
    def __init__(self, map_name: str, output: str, verbose: int = 3):
        super(Sc2Env, self).__init__()

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.redis_client.flushall()

        self.output = output
        self.replays_path = os.path.join(output, "replays")
        os.makedirs(self.replays_path, exist_ok=True)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=Action.number_of_actions() - 1,
            name="action",
        )
        self._observation_spec = {
            "structures_state": array_spec.BoundedArraySpec(
                shape=(224, 224, 3),
                dtype=np.float32,
                minimum=0,
                maximum=255,
                name="structures_state",
            ),
            "units_state": array_spec.BoundedArraySpec(
                shape=(224, 224, 3),
                dtype=np.float32,
                minimum=0,
                maximum=255,
                name="units_state",
            ),
            "minerals": array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.float32, minimum=0, maximum=np.inf, name="minerals"
            ),
            "vespene": array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.float32, minimum=0, maximum=np.inf, name="vespene"
            ),
            "supply_used": array_spec.BoundedArraySpec(
                shape=(1,),
                dtype=np.float32,
                minimum=0,
                maximum=200,
                name="supply_used",
            ),
            "supply_cap": array_spec.BoundedArraySpec(
                shape=(1,),
                dtype=np.float32,
                minimum=0,
                maximum=200,
                name="supply_cap",
            ),
        }
        self._state = {
            "structures_state": np.zeros((224, 224, 3), dtype=np.uint8),
            "units_state": np.zeros((224, 224, 3), dtype=np.uint8),
            "minerals": 0,
            "vespene": 0,
            "supply_used": 0,
            "supply_cap": 0,
        }
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
        self.game_output = os.path.join(
            self.replays_path, f"Artanis-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        os.makedirs(self.game_output, exist_ok=True)

        self.acmrwd = 0.0
        self._state = {
            "structures_state": np.zeros((224, 224, 3), dtype=np.uint8),
            "units_state": np.zeros((224, 224, 3), dtype=np.uint8),
            "minerals": 0,
            "vespene": 0,
            "supply_used": 0,
            "supply_cap": 0,
        }
        self._episode_ended = False
        self.game_status = game_info.GameResult.PLAYING

        if hasattr(self, "game_process") and self.game_process.poll() is None:
            self.game_process.kill()
            self.game_process.wait()

        if platform.system() == "Windows":
            self.game_process = subprocess.Popen(
                [
                    ".venv/Scripts/python.exe",
                    "sc2_rl/sc2/artanis_bot.py",
                ],
            )
        elif platform.system() == "Linux":
            self.game_process = subprocess.Popen(
                [
                    "python",
                    "sc2_rl/sc2/artanis_bot.py",
                ],
            )

        time.sleep(5)

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

        self._state = state_rwd_action["state"]
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
                "structures map",
                cv2.flip(
                    cv2.resize(
                        np.array(
                            self._state["structures_state"],
                            dtype=np.uint8,
                        ),
                        None,
                        fx=4,
                        fy=4,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    0,
                ),
            )
            cv2.imshow(
                "units map",
                cv2.flip(
                    cv2.resize(
                        np.array(self._state["units_state"], dtype=np.uint8),
                        None,
                        fx=4,
                        fy=4,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    0,
                ),
            )
            cv2.waitKey(1)

        if self.verbose >= 3:
            cv2.imwrite(
                f"{self.game_output}/structures-{self.game_tick}.png",
                np.array(
                    self._state["structures_state"],
                    dtype=np.uint8,
                ),
            )
            cv2.imwrite(
                f"{self.game_output}/units-{self.game_tick}.png",
                np.array(self._state["units_state"], dtype=np.uint8),
            )

        if self.verbose >= 1:
            info = state_rwd_action["info"]
            if (
                self.game_tick is not None and self.game_tick % 100 == 0
            ) or self._episode_ended:
                print(
                    f"Game Tick: {self.game_tick}. Total reward: {self.acmrwd:.4f}. Void Ray: {info['n_voidray']}. Nexus: {info['n_nexus']}"
                )

        return (
            ts.termination(self._state, reward)
            if self._episode_ended
            else ts.transition(self._state, reward, discount=0.99)
        )

    def _calculate_macro_reward(self, info) -> float:
        reward = 0

        reward -= 0.0001  # Add more 0?

        if not hasattr(self, "prev_nexus"):
            self.prev_nexus = info["n_nexus"]

        if info["n_nexus"] < self.prev_nexus:
            reward -= 10

        if not hasattr(self, "prev_voidray"):
            self.prev_voidray = info["n_voidray"]

        if info["n_voidray"] < self.prev_voidray:
            reward -= 0.5

        if not hasattr(self, "prev_structures"):
            self.prev_structures = info["n_structures"]

        if info["n_structures"] < self.prev_structures:
            reward -= 0.25

        if self.game_status == game_info.GameResult.VICTORY:
            reward += rwd.GAME_REWARD.WIN
        elif self.game_status == game_info.GameResult.DEFEAT:
            reward += rwd.GAME_REWARD.LOSE
            shutil.rmtree(self.game_output)

        return reward


def preprocess_observation(observation):
    processed_observation = {}
    for key, value in observation.items():
        if key in ["structures_state", "units_state"]:  # Imágenes
            # Redimensionar las imágenes al tamaño esperado y normalizar
            resized_image = tf.image.resize(value, [224, 224])
            normalized_image = resized_image / 255.0
            processed_observation[key] = normalized_image
        else:  # Valores escalares
            # Convertir a tensor y expandir dimensiones si es necesario
            scalar_tensor = tf.expand_dims(
                tf.convert_to_tensor(value, dtype=tf.float32), -1
            )
            # No expandir las dimensiones si ya has ajustado las specs para incluir la dimensión de lote
            processed_observation[key] = scalar_tensor
    return processed_observation


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


def create_environment(map_name: str, output: str, verbose: int):
    os.makedirs(output, exist_ok=True)
    return PreprocessEnvironmentWrapper(Sc2Env(map_name, output, verbose))
