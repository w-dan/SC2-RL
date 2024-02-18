# essentials
import tensorflow as tf
import numpy as np

# custom
from rl.sc2env import Sc2Env
from bots.artanis_bot import ArtanisBot          # TODO: import properly (can't resolve path)

# policies
from tf_agents.policies import random_tf_policy     # testing random as baseline
from tf_agents.specs import array_spec
from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy
from gym.spaces import Discrete

# layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


# environment = Sc2Env()
action_space = Discrete(2)  


class RandomPolicy():
    def __init__(self, environment):
        self.environment = environment
        # self.policy = policy

    def run(self, n_iters):
        time_step = self.environment.reset()  # first time step

        for i in range(n_iters):
            action_step = self.policy.action(time_step)
            next_time_step = self.environment.step(action_step)
            print(f"[+] Action step: {action_step}")
            time_step = next_time_step
            print(f"[+] Iteration {i}")



class DoubleDeepQPolicy():

    """      TODO: find out how to load models
    @staticmethod
    def create_network(input_shape, conv_blocks, dense_blocks, output_neurons):
        model = Sequential()

        # convolutional blocks
        for filters, kernel_size, strides in conv_blocks:
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', input_shape=input_shape))
        
        model.add(Flatten())

        # dense blocks
        for units in dense_blocks:
            model.add(Dense(units=units, activation='relu'))

        # output
        model.add(Dense(units=output_neurons, activation='softmax'))

        return model
    """

    @staticmethod
    def get_inference_result(model, input_data):
        predictions = model.predict(input_data)
        rounded_predictions = np.round(predictions)
        inference_result = int(rounded_predictions[0][0])

        return inference_result
