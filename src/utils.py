import logging
from dotenv import load_dotenv

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


def parse_random_seeds_list(s):
    """
    Converts a comma-separated string of integers into a list of ints.
    Example: "100,200,300..." -> [100, 200, 300, ...]
    """
    return [int(x.strip()) for x in s.strip().split(',')]


def get_seeds_from_env_or_else_default() -> list[int]:
    """
    Retrieves a list of random seeds from the environment variable 'SEEDS'.
    If the environment variable is not set, it returns a default list of seeds.
    """
    import os

    load_dotenv()
    seeds_str = os.getenv('SEEDS', '100,200,300')
    return parse_random_seeds_list(seeds_str)


def connect_to_db() -> tuple:
    """
    Connects to the PostgreSQL database and returns a tuple
    (connection, connection_cursor).

    example: conn, cursor = connect_to_db()
    """
    import psycopg2
    import os

    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("POSTRGES_HOST", "localhost"),
        user=os.getenv("POSTRGES_USER", "postgres"),
        port=os.getenv("POSTGRES_MAPPED_PORT", "5432"),
        password=os.getenv("POSTRGES_PASSWORD", "postgres"),
        database=os.getenv("POSTRGES_DB", "postgres"),
    )
    return conn, conn.cursor()
