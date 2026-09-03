"""SQLite store: dataset registry + run records. Content hashes make datasets
immutable references — loading verifies bytes, so silent edits are caught."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from evalforge.models import Base


def store_path() -> str:
    root = os.environ.get("EVALFORGE_STORE", "./.evalforge")
    os.makedirs(root, exist_ok=True)
    return root


def db_path() -> str:
    return os.path.join(store_path(), "evalforge.db")


def create_app_engine(database_url: str | None = None) -> Engine:
    url = database_url or f"sqlite:///{db_path()}"
    if ":memory:" in url:
        engine = create_engine(
            url, connect_args={"check_same_thread": False, "timeout": 30}, poolclass=StaticPool
        )
    else:
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
