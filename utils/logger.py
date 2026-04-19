import os
import logging
from datetime import datetime
import warnings

class Logger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_dir = os.path.join(self.exp_dir, 'LOG')
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')

        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with both file and console output."""
        logger = logging.getLogger(f'GAR_logger_{datetime.now().strftime("%H%M%S")}')
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (for visibility in terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def info(self, message):
        """Log info message safely."""
        try:
            self.logger.info(message)
        except Exception as e:
            warnings.warn(f"Logging failed: {e}")
            print(f"[INFO-Fallback]: {message}")

    def warning(self, message):
        try:
            self.logger.warning(message)
        except:
            print(f"[WARNING-Fallback]: {message}")

    def error(self, message):
        try:
            self.logger.error(message)
        except:
            print(f"[ERROR-Fallback]: {message}")

    def debug(self, message):
        try:
            self.logger.debug(message)
        except:
            pass  # debug logs may fail silently

def setup_logging(exp_dir):
    """Convenience function to initialize the logger."""
    return Logger(exp_dir)
