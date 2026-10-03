import logging
import sys
from urllib.parse import urlsplit, urlunsplit


def mask_database_url(url: str | None) -> str:
    """
    Safely mask username and password in database connection URLs.
    Example: postgresql://postgres:secret@host:5432/railway -> postgresql://postgres:***@host:5432/railway
    """
    if not url:
        return "<NOT_SET>"
    try:
        parsed = urlsplit(url)
        if not parsed.netloc:
            return "<INVALID_URL_FORMAT>"

        # Parse user:pass@host:port
        user_info = ""
        host_info = parsed.netloc

        if "@" in host_info:
            credentials, host_info = host_info.rsplit("@", 1)
            if ":" in credentials:
                user, _ = credentials.split(":", 1)
                user_info = f"{user}:***@"
            else:
                user_info = f"{credentials}:***@"

        masked_netloc = f"{user_info}{host_info}"
        return urlunsplit((parsed.scheme, masked_netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return "<MASKED_DATABASE_URL>"


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Configure application-wide structured logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger = logging.getLogger("outlet_verification")
    logger.setLevel(level)
    return logger
