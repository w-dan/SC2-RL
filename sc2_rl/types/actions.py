from enum import IntEnum


class Action(IntEnum):
    """
    Defines actions for a strategy game context.

    Attributes:
        - NO_ACTION (0): No action.
        - BUILD_PYLON (1): Build pylon (basic action to increase supply).
        - EXPAND (2): Expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3).
        - BUILD_STARGATE (3): Build stargate (or up to one) (evenly).
        - TRAIN_VOIDRAY (4): Build voidray (evenly).
        - SCOUT (5): Send scout (evenly/random/closest to enemy?).
        - ATTACK (6): Attack (known buildings, units, then enemy base, just go in logical order.).
        - FLEE (7): Voidray flee (back to base).
    """

    NO_ACTION = 0
    BUILD_PYLON = 1
    EXPAND = 2
    BUILD_STARGATE = 3
    TRAIN_VOIDRAY = 4
    SCOUT = 5
    ATTACK = 6
    FLEE = 7

    @classmethod
    def _missing_(cls, value):
        """
        Returns a default action for undefined values.

        Args:
            value: The value being searched for in the enum.

        Returns:
            Action.NO_ACTION: The default action if the value is not found.
        """
        return cls.NO_ACTION

    @classmethod
    def number_of_actions(cls):
        """
        Returns the total number of defined actions in the enum.

        Returns:
            int: The count of actions.
        """
        return len(cls)

    def __eq__(self, other):
        """
        Overrides the default implementation of the equality operator (==) to compare
        an instance of Action with another instance or integer value.

        Args:
            other (Action|int): The object to compare with this instance. Can be another
                                instance of Action or an integer representing the value
                                of an Action.

        Returns:
            bool: True if the other object is equal to this instance, False otherwise.

        Examples:
            >>> Action.BUILD_PYLON == Action.BUILD_PYLON
            True
            >>> Action.BUILD_PYLON == 1
            True
            >>> Action.BUILD_PYLON == Action.EXPAND
            False
        """
        if isinstance(other, IntEnum):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return NotImplemented
