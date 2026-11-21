"""
Database layer (plain SQLAlchemy, no Flask).

`db` keeps the familiar declarative API used by the models and services
(`db.Model`, `db.Column`, `db.relationship`, `db.session`, `Model.query`, `db.func`),
so the data layer is framework-independent. Each HTTP request gets its own SQLAlchemy
session, scoped by a context variable that FastAPI copies into worker threads.
"""

import contextvars
import uuid
from contextlib import contextmanager

import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.orm import DeclarativeBase, Query, relationship, scoped_session, sessionmaker

_scope = contextvars.ContextVar('db_scope', default='startup')


class AppQuery(Query):
    """Query with the 404 helpers the routes use."""

    def first_or_404(self):
        obj = self.first()
        if obj is None:
            raise HTTPException(status_code=404, detail='Not found')
        return obj

    def get_or_404(self, ident):
        obj = self.get(ident)
        if obj is None:
            raise HTTPException(status_code=404, detail='Not found')
        return obj


class _Database:
    # column types and helpers re-exported for model definitions
    Column, Integer, String, Text = sa.Column, sa.Integer, sa.String, sa.Text
    Float, Boolean, DateTime, ForeignKey = sa.Float, sa.Boolean, sa.DateTime, sa.ForeignKey
    func = sa.func
    relationship = staticmethod(relationship)

    def __init__(self):
        self.engine = None
        self._factory = sessionmaker(query_cls=AppQuery, expire_on_commit=False)
        self.session = scoped_session(self._factory, scopefunc=_scope.get)

        class Model(DeclarativeBase):
            query = self.session.query_property(query_cls=AppQuery)

        self.Model = Model

    def init(self, url):
        """Create the engine. SQLite: WAL journal + 30 s busy timeout for concurrent requests."""
        connect_args = {'timeout': 30, 'check_same_thread': False} if url.startswith('sqlite') else {}
        self.engine = sa.create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        if url.startswith('sqlite'):
            @event.listens_for(self.engine, 'connect')
            def _sqlite_pragmas(dbapi_conn, _record):
                cur = dbapi_conn.cursor()
                cur.execute('PRAGMA journal_mode=WAL')
                cur.execute('PRAGMA busy_timeout=30000')
                cur.close()
        self._factory.configure(bind=self.engine)

    def create_all(self):
        self.Model.metadata.create_all(self.engine)

    @contextmanager
    def request_scope(self):
        """One session per request (or background job); always closed at the end."""
        token = _scope.set(uuid.uuid4().hex)
        try:
            yield self.session
        finally:
            self.session.remove()
            _scope.reset(token)


db = _Database()
