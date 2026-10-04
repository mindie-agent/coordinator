"""Initialization and opening of the two local authoritative SQLite stores.

An ownership marker records creation intent only. It cannot restore native
attachments, remote-operation receipts or leases from a lost database.
"""
from __future__ import annotations

import contextlib
import errno
import os
import re
import sqlite3
import time
from pathlib import Path


RECORDS_SCHEMA = (
    "CREATE TABLE records(kind TEXT, id TEXT, data TEXT NOT NULL, PRIMARY KEY(kind,id))",
)
POOL_SCHEMA = RECORDS_SCHEMA + (
    "CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, data TEXT NOT NULL)",
)


class AuthorityStateError(RuntimeError):
    """Required local authority is unavailable; no empty recovery was attempted."""


@contextlib.contextmanager
def initialization_lock(path):
    # The file remains in place: unlinking it could give another process a
    # different inode to lock. Neither age nor a saved PID establishes liveness.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    time.sleep(.01)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Closing releases the OS lock even on interruption.
        os.close(fd)


def write_marker(path, value):
    # Write and fsync before SQLite is allowed to create anything. A short or
    # interrupted marker remains a failure, not permission to start again.
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def table_shapes(db):
    tables = db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return {name: (list(db.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')),
                   bool(re.search(r"\bAUTOINCREMENT\b", sql, re.I))) for name, sql in tables}


class LocalAuthority:
    def __init__(self, path: Path, *, identity: str, schema: tuple[str, ...], residue=(), timeout=10):
        self.path = path
        self.marker = path.with_name(path.name + ".owner")
        self.identity = identity + "\n"
        self.timeout = timeout
        with contextlib.closing(sqlite3.connect(":memory:")) as model:
            for statement in schema:
                model.execute(statement)
            self.shapes = table_shapes(model)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with initialization_lock(path.with_name(path.name + ".init.lock")):
            self._check_marker(required=False)
            fresh = not path.exists()
            if fresh:
                remnants = [path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal")]
                remnants.extend(path.parent / name for name in residue)
                if self.marker.exists() or any(p.exists() or p.is_symlink() for p in remnants):
                    raise self._error("database is missing beside existing state")
                write_marker(self.marker, self.identity)
                with contextlib.closing(sqlite3.connect(path, timeout=timeout)) as db:
                    with db:
                        db.execute("BEGIN IMMEDIATE")
                        for statement in schema:
                            db.execute(statement)
                        db.execute("PRAGMA user_version=1")
            self._check_file()
            with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=timeout)) as db:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    self._validate(db)
                    if not self.marker.exists():
                        # Existing complete legacy stores are adopted without
                        # replacing schema, identities or operation outcomes.
                        write_marker(self.marker, self.identity)
                db.execute("PRAGMA journal_mode=WAL")
        self._authority_identity = self._current_identity()

    def _current_identity(self):
        return tuple((info.st_dev, info.st_ino) for info in (self.path.lstat(), self.marker.lstat()))

    def _error(self, reason):
        return AuthorityStateError(f"authoritative state {self.path}: {reason}; existing state was preserved, not rebuilt")

    def _check_marker(self, *, required=True):
        if self.path.is_symlink() or self.marker.is_symlink():
            raise self._error("database and ownership marker must be regular local files")
        if not self.marker.exists():
            if required:
                raise self._error("ownership marker is missing")
            return
        if not self.marker.is_file() or self.marker.read_text(encoding="utf-8") != self.identity:
            raise self._error("ownership marker is invalid")

    def _check_file(self):
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise self._error("database is missing, empty or invalid")

    def _validate(self, db):
        if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
            raise self._error("unsupported schema version")
        if table_shapes(db) != self.shapes:
            raise self._error("required table or column schema is missing or incompatible")

    @contextlib.contextmanager
    def transaction(self, *, read_only=False):
        self._check_marker()
        self._check_file()
        if self._current_identity() != self._authority_identity:
            raise self._error("database or marker was replaced while this store was in use")
        mode = "ro" if read_only else "rw"
        with contextlib.closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=" + mode, uri=True,
                                               timeout=self.timeout)) as db:
            with db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
                self._validate(db)
                yield db
