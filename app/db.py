from sqlalchemy import create_engine

from .config import settings

engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=10,
    pool_pre_ping=True,
)
