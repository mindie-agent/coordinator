"""Host mail integration for the existing local coordinator.

Only hosts already used by this task participate. No fleet scan, scheduler,
background wakeup or message-driven execution is implemented here.
"""
from __future__ import annotations

import hashlib
import json
import time
import threading
import uuid

MESSAGE_POLL_SECONDS = 5.0
MESSAGE_BATCH_SIZE = 2


def _key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class TaskMessages:
    def _message_hosts(self, store, user, session_id):
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
            if session.get("user") and session["user"] != user:
                raise PermissionError("message task belongs to another user")
            saved = [row for row in store.rows(db, "mailbox")
                     if row["user"] == user and row["session_id"] == session_id]
        hosts = {row["host_id"]: row["endpoint"] for row in saved}
        for row in store.executions(session_id):
            if row.get("user") and row["user"] != user:
                continue
            bindings = [row.get("binding"), *(role.get("binding") for role in row.get("roles") or [])]
            for binding in bindings:
                endpoint = (binding or {}).get("host_endpoint")
                if endpoint and endpoint.get("host"):
                    # Root/cwd are transport details, not a second mailbox.
                    endpoint = {"host": endpoint["host"], "port": endpoint.get("port", 22),
                                "user": endpoint.get("user", "root")}
                    hosts[_key(endpoint)] = endpoint
        return hosts

    def notifications(self, sessions_dir, user, session_id):
        """Return cached mail and start at most one bounded polling worker.

        This call never waits for a host. There is no background client wakeup;
        a later normal call delivers newly cached messages. Delivery means the
        local coordinator returned the record, not that an Agent accepted it.
        """
        if getattr(self.backend, "supports_task_messages", False) is not True or self._mail_stopped.is_set():
            return {}
        store = self.store(sessions_dir)
        hosts = self._message_hosts(store, user, session_id)
        if not hosts:
            return {}
        waiting = any(row.get("phase") in {"queued", "waiting", "waiting_for_runtime"}
                      for row in store.executions(session_id))
        now = time.time()
        with store.transaction() as db:
            mailboxes = {row["host_id"]: row for row in store.rows(db, "mailbox")
                         if row["user"] == user and row["session_id"] == session_id}
            cached = [row for row in store.rows(db, "mail-message")
                      if row.get("mail_user") == user and row.get("mail_session") == session_id
                      and not row.get("delivered_at")][:MESSAGE_BATCH_SIZE]
            for row in cached:
                row["delivered_at"] = now
                store.put(db, "mail-message", row)
        internal = {"id", "host_id", "mail_user", "mail_session", "delivered_at",
                    "requested_recipient", "requested_session"}
        result = {}
        if cached:
            result["notifications"] = [{key: value for key, value in row.items() if key not in internal}
                                       for row in cached]
        peers = [peer for row in mailboxes.values() for peer in row.get("peers", [])]
        if waiting and peers:
            result["coordination_peers"] = peers[:5]
        failures = [{"host": row["host_id"], "error": row["error"]}
                    for row in mailboxes.values() if row.get("error")]
        if failures:
            result["notification_status"] = {"state": "unavailable", "hosts": failures[:5]}
        due = {key: endpoint for key, endpoint in hosts.items()
               if now - mailboxes.get(key, {}).get("checked_at", 0) >= MESSAGE_POLL_SECONDS}
        if due:
            lock = self._lock_for("mailbox", _key([str(store.state_dir), user, session_id]))
            if lock.acquire(blocking=False):
                worker = threading.Thread(target=self._poll_notifications,
                                          args=(store, user, session_id, due, waiting, lock),
                                          name="mindie-mail-" + session_id[-12:], daemon=True)
                try:
                    with self._registry_guard:
                        if self._mail_stopped.is_set():
                            lock.release()
                            return result
                        self._mail_workers.add(worker)
                        worker.start()
                except Exception:
                    with self._registry_guard:
                        self._mail_workers.discard(worker)
                    lock.release()
                    raise
        return result

    def _poll_notifications(self, store, user, session_id, hosts, waiting, lock):
        try:
            remaining = MESSAGE_BATCH_SIZE
            for host_id, endpoint in hosts.items():
                if not remaining or self._mail_stopped.is_set():
                    break
                mailbox_id = _key([user, session_id, host_id])
                with store.transaction() as db:
                    try:
                        mailbox = store.get(db, "mailbox", mailbox_id)
                    except ValueError:
                        mailbox = {"id": mailbox_id, "user": user, "session_id": session_id,
                                   "host_id": host_id, "endpoint": endpoint, "cursor": 0}
                if time.time() - mailbox.get("checked_at", 0) < MESSAGE_POLL_SECONDS:
                    continue
                mailbox["checked_at"] = time.time()
                try:
                    response = self.backend.host({"host_endpoint": endpoint}, {
                        "action": "message-events", "user": user, "session_id": session_id,
                        "after": mailbox["cursor"], "mailbox_epoch": mailbox.get("epoch"),
                        "limit": remaining, "include_peers": waiting})
                    if self._mail_stopped.is_set():
                        return
                    if response.get("status") != "ok":
                        raise RuntimeError(response.get("error") or "host messages unavailable")
                    received = [{**event, "reply_reference": {"host": host_id, "message_id": event["message_id"]}}
                                for event in response["events"]]
                    updated = {**mailbox, "checked_at": time.time(), "cursor": response["cursor"], "epoch": response["mailbox_epoch"],
                               "peers": [{"reference": {"host": host_id, "user": peer["user"],
                                                         "session_id": peer["session_id"]},
                                          "seen_at": peer["seen_at"]} for peer in response.get("peers", [])]}
                    updated.pop("error", None)
                    # Persist inbox text and cursor atomically. Remote receipt
                    # is not Agent acknowledgement; full records stay queryable
                    # under mail-message even after automatic local delivery.
                    with store.transaction() as db:
                        for event in received:
                            store.put(db, "mail-message", {"id": _key([host_id, updated["epoch"], event["message_id"]]),
                                                          "host_id": host_id, "mail_user": user,
                                                          "mail_session": session_id, **event})
                        store.put(db, "mailbox", updated)
                    remaining -= len(received)
                except Exception as exc:
                    if self._mail_stopped.is_set():
                        return
                    mailbox["checked_at"] = time.time()
                    mailbox["error"] = str(exc)[:300]
                    with store.transaction() as db:
                        store.put(db, "mailbox", mailbox)
        finally:
            lock.release()
            with self._registry_guard:
                self._mail_workers.discard(threading.current_thread())

    def close_messages(self):
        """Quiesce optional workers before an embedded service/store is removed."""
        self._mail_stopped.set()
        with self._registry_guard:
            workers = list(self._mail_workers)
        deadline = time.monotonic() + 60
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))

    def message(self, sessions_dir, user, session_id, recipient, text):
        if getattr(self.backend, "supports_task_messages", False) is not True:
            raise ValueError("this backend does not support task messages")
        if self._mail_stopped.is_set():
            raise RuntimeError("coordinator messages are closing")
        if not isinstance(recipient, dict) or not isinstance(recipient.get("host"), str):
            raise ValueError("use a coordination reference returned by run/status or a reply_reference")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("message must contain 1..4000 characters")
        reply_to = recipient.get("message_id")
        expected = {"host", "message_id"} if reply_to else {"host", "user", "session_id"}
        if set(recipient) != expected or any(not isinstance(value, str) or not value for value in recipient.values()):
            raise ValueError("invalid coordination reference")
        store = self.store(sessions_dir)
        hosts = self._message_hosts(store, user, session_id)
        endpoint = hosts.get(recipient["host"])
        if endpoint is None:
            raise ValueError("message host has not been used by this task")
        intent_id = _key([user, session_id, recipient, text])
        with self._lock_for("message-send", intent_id):
            with store.transaction() as db:
                try:
                    pending = store.get(db, "mail-outbox", intent_id)
                except ValueError:
                    pending = {"id": intent_id, "message_id": uuid.uuid4().hex}
                    store.put(db, "mail-outbox", pending)
            request = {"action": "reply" if reply_to else "message", "user": user,
                       "session_id": session_id, "message_id": pending["message_id"], "text": text}
            if reply_to:
                request["reply_to"] = reply_to
            else:
                request.update(recipient=recipient["user"], recipient_session=recipient["session_id"])
            response = self.backend.host({"host_endpoint": endpoint}, request)
            if response.get("status") != "sent":
                raise RuntimeError(response.get("error") or "message was not stored")
            with store.transaction() as db:
                store.put(db, "mail-sent", {"id": pending["message_id"], "host_id": recipient["host"],
                                           **response["message"]})
                db.execute("DELETE FROM records WHERE kind='mail-outbox' AND id=?", (intent_id,))
            return {"state": "sent", "message": response["message"]}
