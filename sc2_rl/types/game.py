from enum import IntEnum

SCOUT_TIMEOUT = 200
MIN_VOIDRAY = 1
MIN_NEXUS = 1


class MAX_WORKERS:
    NEXUS = 16
    VESPENE_GEYSER = 3


class RANGES:
    BUILD = 10
    ATTACK = 10
    REWARD = 8


class GameResult(IntEnum):
    PLAYING = 0
    VICTORY = 1
    DEFEAT = 2
    TIE = 3

    def __eq__(self, other):
        """
        Overrides the default implementation of the equality operator (==) to compare
        an instance of GameResult with another instance or integer value.

        Args:
            other (GameResult|int): The object to compare with this instance. Can be another
                                instance of GameResult or an integer representing the value
                                of an GameResult.

        Returns:
            bool: True if the other object is equal to this instance, False otherwise.

        Examples:
            >>> GameResult.PLAYING == GameResult.PLAYING
            True
            >>> GameResult.VICTORY == 1
            True
            >>> GameResult.DEFEAT == GameResult.TIE
            False
        """
        if isinstance(other, IntEnum):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return NotImplemented
