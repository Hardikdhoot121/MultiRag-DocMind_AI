import functools
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("docmind.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger("DocMind")

class DocMindError(Exception):
    pass

class FileProcessingError(DocMindError):
    pass

class EmbeddingError(DocMindError):
    pass

class VectorStoreError(DocMindError):
    pass

def handle_errors(fallback=None, reraise=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DocMindError:
                raise
            except Exception as e:
                logger.error(f"{func.__name__}() failed: {type(e).__name__}: {e}")
                if reraise:
                    raise DocMindError(f"{func.__name__} failed: {e}") from e
                return fallback
        return wrapper
    return decorator
