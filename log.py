import logging
from enum import Enum


class LoggerOption(Enum):
    SCREEN = 1
    FILE = 2

class CustomLogger:
    def __init__(self, output: LoggerOption = LoggerOption.SCREEN, log_file: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] : %(message)s')
        
        if output == LoggerOption.SCREEN:
            self._add_screen_handler(formatter)
        elif output == LoggerOption.FILE:
            if log_file:
                self._add_file_handler(formatter, f"logs/{log_file}.log")
            else:
                raise ValueError("Log file path is required when output is 'file'.")
        else:
            raise ValueError("Invalid output option. Choose between 'screen' or 'file'.")

    def _add_screen_handler(self, formatter):
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        self.logger.addHandler(screen_handler)

    def _add_file_handler(self, formatter, log_file):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# Example usage:
if __name__ == "__main__":
    logger = CustomLogger(output='screen')  # Log to screen
    logger.info("This is an informational message.")
    logger.error("This is an error message.")

    logger = CustomLogger(output='file', log_file='example.log')  # Log to file
    logger.info("This is an informational message.")
    logger.error("This is an error message.")
