from typing import Dict, List

from sc2_rl.types.actions import Action


class GAME_REWARD:
    WIN = 500
    LOSE = -500


class BUILD_REWARD:
    NEXUS = 5
    STARGATE = 1.5
    CYBERNETICSCORE = 1
    GATEWAY = 1
    PYLON = 0.2
    ASSIMILATOR_NOT_BUILT = -0.2


class TROOPS_REWARD:
    PROBE_SCOUTING_STATIC = 0.2  # DEBUG
    VOIDRAY_TRAINED = 0.1
    PROBE_SCOUTING = 0.002
    MAX_PROBES = -0.05
    PROBE_DESTROY_SCOUTING = -0.15
    NO_VOIDRAY_ATTACK = -0.2


class MicroReward:
    """
    Represents a small reward given for a specific action at a certain game tick.

    Attributes:
        game_tick (int): The game tick at which the action occurred.
        action (Action): The action performed that led to the reward.
        reward (float): The numerical value of the reward.
    """

    def __init__(self, game_tick: int, action: Action, reward: float):
        self.game_tick = game_tick
        self.action = action
        self.reward = reward


class RewardManager:
    """
    Manages rewards for different units in a game, tracking rewards by unit tags.

    Attributes:
        rewards (Dict[int, List[MicroReward]]): A dictionary mapping unit tags to lists of MicroRewards.
        next_tick_rewards (float): The total rewards to be applied in the next tick.
    """

    def __init__(self):
        self.rewards: Dict[int, List[MicroReward]] = {}
        self.next_tick_rewards = 0

    def add_reward(self, unit_tag: int, reward: MicroReward):
        """
        Adds a reward for a specific unit.

        Args:
            unit_tag (int): The tag of the unit to which the reward is assigned.
            reward (MicroReward): The reward to be added.
        """
        if unit_tag not in self.rewards:
            self.rewards[unit_tag] = []
        self.rewards[unit_tag].append(reward)

    def consume_rewards(self, unit_tag: int, consume=True):
        """
        Consumes (calculates and optionally clears) rewards for a specific unit.

        Args:
            unit_tag (int): The tag of the unit whose rewards are to be consumed.
            consume (bool, optional): Whether to remove the unit's rewards after consuming. Defaults to True.
        """
        if unit_tag in self.rewards:
            total_unit_reward = 0
            for micro_reward in self.rewards[unit_tag]:
                total_unit_reward += micro_reward.reward
                print(
                    f"Action '{micro_reward.action.name}' from game tick {micro_reward.game_tick} received {micro_reward.reward:.4f} reward"
                )

            self.next_tick_rewards += total_unit_reward

            if consume:
                del self.rewards[unit_tag]

    def add_consume_reward(self, unit_tag: int, reward: MicroReward):
        """
        Adds a reward for a specific unit and immediately consumes it.

        Args:
            unit_tag (int): The tag of the unit to which the reward is assigned.
            reward (MicroReward): The reward to be added and consumed.
        """
        self.add_reward(unit_tag, reward)
        self.consume_rewards(unit_tag)

    def apply_scout_reward(self, scout_list):
        """
        Applies and consumes rewards for scouts, based on their activity.

        Args:
            scout_list (list): A list of tuples (scout_tag, is_idle) indicating which scouts are active or idle.
        """
        for scout, idle in scout_list:
            self.consume_rewards(scout, consume=idle)

    def apply_scout_destroy(self, scout):
        """
        Applies a specific reward for a scout being destroyed and consumes it.

        Args:
            scout (int): The tag of the scout that was destroyed.
        """
        if scout in self.rewards:
            for micro_reward in self.rewards[scout]:
                micro_reward.reward = TROOPS_REWARD.PROBE_DESTROY_SCOUTING

            self.consume_rewards(scout)

    def apply_next_tick_rewards(self):
        """
        Applies the total accumulated rewards for the next tick and resets the counter.

        Returns:
            float: The total accumulated rewards.
        """
        accumulated_rewards = self.next_tick_rewards
        self.next_tick_rewards = 0

        return accumulated_rewards

    def apply_unsuccessfull_action(self, reward: MicroReward):
        print(
            f"Try action '{reward.action.name}' unsuccessfully from game tick {reward.game_tick} received {reward.reward:.4f} reward"
        )
        self.next_tick_rewards += reward.reward
