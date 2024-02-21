# essentials
import numpy as np

# policies
from tf_agents.policies.random_tf_policy import RandomTFPolicy

# custom
# from sc2env import Sc2Env

# # layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten


class RandomPolicy:

    def __init__(self, time_step_spec, action_spec):
        self.policy = RandomTFPolicy(time_step_spec, action_spec)

    def take_action(self, time_step):
        # random action
        action_step = self.policy.action(time_step)
        action = action_step.action.numpy()[0]
        print(f"[+] Taking action: {action}")

        return action


class DoubleDeepQPolicy:
    """TODO: find out how to load models
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
