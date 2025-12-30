import logging
import re
import sys
import threading
import traceback
from pathlib import Path

from src.constants import LOGS_FILE_NAME


ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class TeeStream(object):
    """
    A 'T-Connector' for output.
    1. Writes RAW output to the Terminal (instant, colored).
    2. Writes BUFFERED output to the Logger (ensures complete lines, adds timestamps).
    """
    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self._local = threading.local()
        self.raw_buffer = ""

    def write(self, message):
        # 1. RAW TERMINAL (Instant)
        if self.original_stream:
            self.original_stream.write(message)
            self.original_stream.flush()

        if getattr(self._local, 'processing', False):
            return

        # 2. LOG FILE (Buffered Line-by-Line)
        if message:
            try:
                self._local.processing = True

                # Append new text to the buffer
                clean_message = ANSI_ESCAPE.sub('', message)

                # TQDM Filter (Progress bars)
                if self.level == logging.ERROR:
                    # Check for typical tqdm signatures (% progress, iterations per sec)
                    if any(x in clean_message for x in ['%|', 'it/s', 's/it']):
                        # Ignore tqdm output
                        self._local.processing = False
                        return

                self.raw_buffer += clean_message

                # Process ALL complete lines in the buffer
                while '\n' in self.raw_buffer:
                    # Peel off the first complete line
                    line, self.raw_buffer = self.raw_buffer.split('\n', 1)

                    # Log the line if it has content.
                    # We use rstrip() to remove the trailing newline char,
                    # BUT we preserve leading whitespace to keep indentation/arrows aligned.
                    if line.strip():
                        self.logger.log(self.level, line.rstrip())

            except Exception:
                pass
            finally:
                self._local.processing = False

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except ValueError:
                pass

        # Flush any remaining text in the buffer (e.g. text without a final newline)
        if getattr(self._local, 'processing', False):
            return

        try:
            self._local.processing = True
            if self.raw_buffer.strip():
                self.logger.log(self.level, self.raw_buffer.rstrip())
                self.raw_buffer = ""
        except Exception:
            pass
        finally:
            self._local.processing = False

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Catches crashes.
    KeyboardInterrupts go to raw stderr.
    Everything else is manually formatted and sent to sys.stderr (TeeStream).
    """
    if issubclass(exc_type, KeyboardInterrupt):
        if sys.__stderr__ is not None:
            sys.__stderr__.write("\n🛑 Interrupted by user (Ctrl+C). Exiting...\n")
        return

    # 1. Format the traceback manually into a list of strings
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

    # 2. Join them into one string
    tb_string = "".join(tb_lines)

    logging.error("Uncaught exception:")

    # 3. Write to stderr (TeeStream)
    # This triggers TeeStream.write(), which splits the string by line
    # and logs each line individually. This ensures EVERY line gets a timestamp.
    sys.stderr.write(tb_string)

def configure_logging():
    root_logger = logging.getLogger()

    handlers_to_remove = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
    for h in handlers_to_remove:
        root_logger.removeHandler(h)

    sys.stdout = TeeStream(root_logger, logging.INFO, sys.__stdout__)
    sys.stderr = TeeStream(root_logger, logging.ERROR, sys.__stderr__)

    sys.excepthook = handle_exception


def resume_logging(experiment_folder: Path):
    log_path = experiment_folder / LOGS_FILE_NAME

    file_handler = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    # Python defaults to WARNING, so it ignores INFO logs (stdout) unless we change this.
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    print(f"📄 Logging set up: writing to {log_path}")

    configure_logging()

