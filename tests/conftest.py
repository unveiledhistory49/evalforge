"""Shared fixtures: isolated in-memory DB per test module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy.orm import Session

from evalforge.db import create_app_engine, create_session_factory, init_db


@pytest.fixture()
def db() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    init_db(engine)
    session = create_session_factory(engine)()
    yield session
    session.close()
    engine.dispose()
