
from logging.config import dictConfig

def configure_logging(level="INFO"):
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
            }
        },
        "handlers": {
            # Gunicorn reads this stream and writes it to its error log/stdout
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {
            "level": level,
            "handlers": ["wsgi"]
        },
    })
