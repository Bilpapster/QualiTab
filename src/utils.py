import logging

def configure_logging() -> logging.Logger:
    """
    Configures the logging for the application.

    This function sets up the basic configuration for logging,
    specifying the log level, format, and other settings.
    It then returns a logger instance that can be used throughout the application.

    Returns:
        logging.Logger: A logger instance configured with the specified settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s::%(funcName)s::%(lineno)d %(asctime)s - %(levelname)s - %(message)s - '
    )
    return logging.getLogger(__name__)