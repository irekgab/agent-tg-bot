import logging
import os
from logging.handlers import TimedRotatingFileHandler


class SuccessResponseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "200 OK" not in record.getMessage()


def configure_logging() -> None:
    log_dir = ".data"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "agent.log")

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    file_handler = TimedRotatingFileHandler(
        log_file, 
        when="midnight", 
        interval=1, 
        backupCount=30
    )
    file_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.INFO)
    httpx_logger.addFilter(SuccessResponseFilter())
