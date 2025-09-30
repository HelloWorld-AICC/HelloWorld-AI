import logging
import os
import sys
from datetime import datetime


_LOGGING_CONFIGURED = False


class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> None:
        if message and message != "\n":
            self._buffer += message
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.logger.log(self.level, line)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
            self._buffer = ""


def setup_logging(
    force: bool = False, base_dir: str | None = None, redirect_prints: bool = False
) -> None:
    """
    Set up application logging to logs/{timestamp}.log with UTF-8 encoding.

    - If force=True, remove existing handlers and reconfigure.
    - Otherwise, add a UTF-8 FileHandler once per process.
    - base_dir sets the directory where logs/ is created (default: CWD).
    """
    global _LOGGING_CONFIGURED

    if base_dir is None:
        base_dir = os.getcwd()

    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if force:
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        _LOGGING_CONFIGURED = False

    if _LOGGING_CONFIGURED:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="[ %(asctime)s | %(levelname)s ] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if redirect_prints:
        sys.stdout = _StreamToLogger(root_logger, logging.INFO)  # type: ignore
        sys.stderr = _StreamToLogger(root_logger, logging.ERROR)  # type: ignore

    _LOGGING_CONFIGURED = True
