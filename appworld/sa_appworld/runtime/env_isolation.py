"""Run AppWorld tool calls without leaving any trace in the environment.

`EnvIsolation.execute(name, args)` returns a call's output and then puts back every piece of environment state the
call can change, so the environment afterwards is exactly as before. The state:

- the contents of every app database (in-memory SQLite, `CachedDBHandler.cache`);
- each database's change log (`DBChangesTracker.changes`), from which AppWorld saves the "changes" format;
- the per-model write counters (`ModelHashHandler.data`), which evaluation uses to find changed models;
- the environment's interaction counter and I/O log (`env.num_interactions`, `env.environment_io`);
- the requester's log of API requests (`env.requester.request_tracker.requests`), which also enforces the request limit;
- each app's set of logged-out tokens (`logging_manager.blacklisted_tokens`);
- the global `random` generator, which AppWorld seeds per task and the APIs draw from at run time (login token
  expiry, spotify.shuffle_song_queue). A draw by an isolated call would change every later draw of the real run;
- the task's output files on disk: `env.execute` ends by saving the database changes, the model counters and the
  logs (`env._save_state`, `env.save_logs`). During an isolated call both are disabled, so the files keep the state
  of the last real call.

Database contents are restored from clean copies that always equal the live databases between calls. Every write
goes through the app's SQLAlchemy engine, so the change log of a database grows exactly when the database is written:
after an isolated call only the written databases are copied back, and after a real call (`commit_point`) only the
written databases are copied again. A read-only call copies nothing.

Threads: the clean copies are created in one thread and used in others (the Speculator's), so they allow any thread;
the caller must serialize every use of the environment (real calls with their commit_point, and isolated calls).

Failure: if putting the state back fails, `execute` raises `IsolationError` after attempting every other restore
step. The environment may then be dirty, so the caller must stop the task; it must not continue or commit.

`environment_fingerprint(env)` hashes the same state independently of the change logs (raw database bytes, file
contents); scripts/test_preexec_isolation.py uses it to check that isolated calls leave the fingerprint unchanged.
"""

import hashlib
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from typing import Any, Callable, Dict, List

from appworld.apps.model_lib import CachedDBHandler, ModelHashHandler

APP_NAMES = ("admin", "amazon", "api_docs", "file_system", "gmail", "phone", "simple_note", "splitwise", "spotify",
             "supervisor", "todoist", "venmo")


class IsolationError(RuntimeError):
    """The environment could not be put back after an isolated call, or was not clean before it."""


def loaded_logging_managers() -> Dict[str, Any]:
    """The login managers of the app modules this process has imported."""
    managers = {}
    for app_name in APP_NAMES:
        module = sys.modules.get(f"appworld.apps.{app_name}.apis")
        manager = getattr(module, "logging_manager", None) if module is not None else None
        if manager is not None and hasattr(manager, "blacklisted_tokens"):
            managers[app_name] = manager
    return managers


def live_connection(db_path: str) -> sqlite3.Connection:
    engine, _ = CachedDBHandler.cache[db_path]
    return engine.raw_connection().connection


class EnvIsolation:
    """Clean database copies plus the snapshot/restore of the non-database state, for one AppWorld environment.

    Construct it right after the environment is created; call `commit_point()` after every real (non-isolated)
    call, so the clean copies keep equal to the live databases.
    """

    def __init__(self, env, execute: Callable[[Any, str, Dict[str, Any]], str]):
        self.env = env
        self.execute_tool = execute
        self.clean: Dict[str, sqlite3.Connection] = {}
        self.change_counts: Dict[str, int] = {}
        for db_path in CachedDBHandler.cache:
            self.refresh(db_path)

    def refresh(self, db_path: str) -> None:
        # check_same_thread=False: restored from the Speculator's thread; the caller serializes all uses.
        copy = self.clean.get(db_path) or sqlite3.connect(":memory:", check_same_thread=False)
        live_connection(db_path).backup(copy)
        self.clean[db_path] = copy
        self.change_counts[db_path] = len(CachedDBHandler.cache[db_path][1].changes)

    def written(self) -> List[str]:
        return [db_path for db_path, (_, tracker) in CachedDBHandler.cache.items()
                if len(tracker.changes) != self.change_counts.get(db_path)]

    def commit_point(self) -> List[str]:
        """After a real call: copy the databases it wrote. Returns their paths."""
        if set(CachedDBHandler.cache) != set(self.clean):
            raise IsolationError("the set of AppWorld databases changed; EnvIsolation cannot follow it")
        written = self.written()
        for db_path in written:
            self.refresh(db_path)
        return written

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """The call's output, with every change it made to the environment undone; IsolationError if that fails."""
        env = self.env
        if set(CachedDBHandler.cache) != set(self.clean) or self.written():
            raise IsolationError("live databases differ from the clean copies; call commit_point() after real calls")
        model_hashes = {home: {app: Counter(counts) for app, counts in apps.items()}
                        for home, apps in ModelHashHandler.data.items()}
        num_interactions = env.num_interactions
        io_length = len(env.environment_io)
        tracker = env.requester.request_tracker
        requests_length = len(tracker.requests)
        requests_reset_on = tracker.num_requests_reset_on
        blacklists = {app: set(m.blacklisted_tokens) for app, m in loaded_logging_managers().items()}
        random_state = random.getstate()
        # Instance attributes shadow the methods env.execute calls at its end, so nothing is written to disk.
        env._save_state = lambda directory: None
        env.save_logs = lambda: None
        try:
            return self.execute_tool(env, name, args)
        finally:
            failures: List[str] = []

            def attempt(label: str, action: Callable[[], None]) -> None:
                try:
                    action()
                except Exception as error:  # keep restoring the rest, then fail the call
                    failures.append(f"{label}: {error!r}")

            def restore_database(db_path: str) -> None:
                self.clean[db_path].backup(live_connection(db_path))
                del CachedDBHandler.cache[db_path][1].changes[self.change_counts[db_path]:]

            def restore_model_hashes() -> None:
                ModelHashHandler.data.clear()  # in place, so holders of the mapping see the restored one
                for home, apps in model_hashes.items():
                    restored_home = ModelHashHandler.data[home]  # keeps the key even when it maps to no apps
                    for app, counts in apps.items():
                        restored_home[app] = Counter(counts)

            def restore_logs() -> None:
                env.num_interactions = num_interactions
                del env.environment_io[io_length:]
                del tracker.requests[requests_length:]
                tracker.num_requests_reset_on = requests_reset_on

            def restore_blacklists() -> None:
                for app, manager in loaded_logging_managers().items():
                    manager.blacklisted_tokens.clear()
                    manager.blacklisted_tokens.update(blacklists.get(app, set()))

            def restore_disk_saving() -> None:
                del env._save_state
                del env.save_logs

            if set(CachedDBHandler.cache) != set(self.clean):
                failures.append("the set of AppWorld databases changed during the call")
            for db_path in [p for p in self.written() if p in self.clean]:
                attempt(f"database {db_path}", lambda p=db_path: restore_database(p))
            attempt("model hashes", restore_model_hashes)
            attempt("interaction and request logs", restore_logs)
            attempt("token blacklists", restore_blacklists)
            attempt("random state", lambda: random.setstate(random_state))
            attempt("disk saving", restore_disk_saving)
            if failures:
                raise IsolationError("isolated call not undone: " + "; ".join(failures))


def database_bytes_digest(connection: sqlite3.Connection) -> str:
    """Hash of the database's raw bytes, except three SQLite header counters that a backup bumps without changing
    content: the file change counter (bytes 24-27), the schema cookie (40-43) and version-valid-for (92-95)."""
    data = bytearray(connection.serialize())
    for start in (24, 40, 92):
        data[start:start + 4] = b"\0\0\0\0"
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


def directory_digest(directory: str) -> str:
    """Hash of every file's relative path and bytes under directory (missing directory: empty)."""
    digest = hashlib.sha256()
    if os.path.isdir(directory):
        for root, _, files in sorted(os.walk(directory)):
            for file_name in sorted(files):
                path = os.path.join(root, file_name)
                digest.update(os.path.relpath(path, directory).encode() + b"\0")
                with open(path, "rb") as handle:
                    digest.update(handle.read())
    return digest.hexdigest()[:16]


def environment_fingerprint(env) -> Dict[str, str]:
    """A hash per state component listed in the module docstring; databases by raw bytes, not by change logs."""
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]

    fingerprint = {}
    for db_path, (engine, tracker) in sorted(CachedDBHandler.cache.items()):
        fingerprint[f"db:{db_path}"] = database_bytes_digest(engine.raw_connection().connection)
        fingerprint[f"changes:{db_path}"] = digest([[str(s), str(p), bool(m)] for s, p, m in tracker.changes])
    fingerprint["model_hashes"] = digest({home: {app: dict(c) for app, c in apps.items()}
                                          for home, apps in ModelHashHandler.data.items()})
    fingerprint["num_interactions"] = str(env.num_interactions)
    fingerprint["environment_io"] = digest(env.environment_io)
    fingerprint["requests"] = digest([env.requester.request_tracker.requests,
                                      env.requester.request_tracker.num_requests_reset_on])
    fingerprint["blacklists"] = digest({app: sorted(m.blacklisted_tokens)
                                        for app, m in loaded_logging_managers().items()})
    fingerprint["random_state"] = digest(random.getstate())
    fingerprint["disk_dbs"] = directory_digest(env.output_db_home_path_on_disk)
    fingerprint["disk_logs"] = directory_digest(env.output_logs_directory)
    return fingerprint


def fingerprint_differences(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
