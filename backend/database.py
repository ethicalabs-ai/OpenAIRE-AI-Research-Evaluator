import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# python-dotenv is loaded by auth_utils (imported before database.py in server.py).
# We intentionally avoid double-loading here to keep startup fast.

# Allow full DATABASE_URL override for any SQLAlchemy-supported backend
# (e.g. postgresql+psycopg2://... for production).
# Falls back to a local SQLite file so development requires zero configuration.
_default_db_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "collaborative.db",
)
os.makedirs(os.path.dirname(_default_db_path), exist_ok=True)

DATABASE_URL: str = (
    os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{_default_db_path}"
)

# SQLite needs check_same_thread=False; other backends don't accept it.
# timeout=30 lets the CLI wait up to 30s for the server to release the lock.
_connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
