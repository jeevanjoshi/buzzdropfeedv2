import os
import sys
import logging
import json
import traceback
import datetime
from typing import Dict, Any, Optional


class PipelineLogger:
    """
    Enterprise Structured Logger providing dual output:
    1. Rich colored console logs for real-time monitoring.
    2. Persistent JSON structured logs (logs/pipeline.log) for crash diagnostics & automated fix hints.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        log_name = os.getenv("CSVG_LOG_FILENAME", "csvg_execution.log")
        self.log_file = os.path.join(self.log_dir, log_name)

        # Configure Root Logger
        self.logger = logging.getLogger("CSVG_PIPELINE")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Reset handlers

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File Handler (Structured JSON)
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def set_log_dir(self, log_dir: str):
        """Re-route all logging output to a different directory."""
        self.log_dir = log_dir
        log_name = os.getenv("CSVG_LOG_FILENAME", "csvg_execution.log")
        self.log_file = os.path.join(self.log_dir, log_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # Recreate file handler on the new log file path
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def log(
        self,
        level: str,
        stage: str,
        message: str,
        pipeline_id: Optional[str] = None,
        component: str = "SYSTEM",
        extra_data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        fix_hint: Optional[str] = None
    ):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "pipeline_id": pipeline_id or "UNKNOWN",
            "stage": stage,
            "component": component,
            "message": message,
            "extra_data": extra_data or {},
            "fix_hint": fix_hint or None
        }

        if exception:
            log_entry["exception_type"] = type(exception).__name__
            log_entry["exception_message"] = str(exception)
            log_entry["stack_trace"] = traceback.format_exc()

        json_line = json.dumps(log_entry)

        # Console Log
        console_msg = f"[{stage}] [{component}] {message}"
        if fix_hint:
            console_msg += f" | FIX HINT: {fix_hint}"

        if level.upper() == "ERROR" or level.upper() == "CRITICAL":
            self.logger.error(console_msg)
        elif level.upper() == "WARNING":
            self.logger.warning(console_msg)
        else:
            self.logger.info(console_msg)

        # File Log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

    def info(self, stage: str, message: str, **kwargs):
        self.log("INFO", stage, message, **kwargs)

    def warning(self, stage: str, message: str, **kwargs):
        self.log("WARNING", stage, message, **kwargs)

    def error(self, stage: str, message: str, **kwargs):
        self.log("ERROR", stage, message, **kwargs)


# Global Logger Instance
logger = PipelineLogger()
