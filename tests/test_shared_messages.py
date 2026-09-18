"""Independent local coordinators communicate only through the host mailbox."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.host.mindie_npu_coordination import CoordinationError, NpuCoordinator, handle_request
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.service import CoordinatorService
from mindie_coordinator.task_client import TaskClient
from mindie_coordinator.task_messages import _key


class MailHost:
    supports_task_messages = True

    def __init__(self, directory):
        self.directory = directory
        self.calls = []
        self.fail_after_send = False
        self.block = None

    def host(self, runtime, request):
        self.calls.append(copy.deepcopy(request))
        if self.block is not None:
            self.block.wait(5)
        def forbidden():
            raise AssertionError("mail must not probe hardware or ports")
        response = handle_request({**request, "state_dir": str(self.directory)},
                                  probe=forbidden, listening_ports=forbidden)
        if request["action"] == "message" and self.fail_after_send:
            self.fail_after_send = False
            raise TimeoutError("host stored message but response was lost")
        return response

    def catalog(self):
        raise AssertionError("messages must not scan the fleet")


@pytest.fixture
def pair(tmp_path):
    backend = MailHost(tmp_path / "shared-host")
    clients = []
    for user in ("alice", "bob"):
        store = AgentSessions(tmp_path / user / "sessions")
        context = store.attach("codex", user + "-native", str(tmp_path))
        pool = RuntimePool(tmp_path / user / "coordinator", backend)
        service = CoordinatorService(pool.state_dir, pool=pool, backend=backend, sessions=store)
        client = TaskClient(context["context_file"], user=user, service=service)
        row = store.execution(client.context, user + "-run", {})
        row.update(user=user, phase="queued", binding={"host_endpoint": {"host": "shared", "user": "root", "port": 22}})
        store.save_execution(row)
        clients.append(client)
    yield backend, clients
    for client in clients:
        client.coordinator.close_messages()


def join_poll(client):
    key = _key([str(client.store.state_dir), client.user, client.context["session"]["id"]])
    lock = client.coordinator._lock_for("mailbox", key)
    assert lock.acquire(timeout=5), "mail worker did not finish"
    lock.release()


def expire_poll(client):
    with client.store.transaction() as db:
        for row in client.store.rows(db, "mailbox"):
            row["checked_at"] = 0
            client.store.put(db, "mailbox", row)


def poll(client):
    expire_poll(client)
    value = client.status()
    join_poll(client)
    received = client.status()
    if value.get("notifications"):
        received["notifications"] = value["notifications"] + received.get("notifications", [])
    return received


def addresses(pair):
    _, (alice, bob) = pair
    poll(alice)
    bob_view = poll(bob)
    alice_view = poll(alice)
    return (next(peer["reference"] for peer in bob_view["coordination_peers"] if peer["reference"]["user"] == "alice"),
            next(peer["reference"] for peer in alice_view["coordination_peers"] if peer["reference"]["user"] == "bob"))


def test_two_independent_coordinators_deliver_and_reply_without_changing_leases(pair):
    backend, (alice, bob) = pair
    assert alice.store.db_path != bob.store.db_path
    assert alice.coordinator.pool.db_path != bob.coordinator.pool.db_path
    to_alice, to_bob = addresses(pair)
    queue = NpuCoordinator(backend.directory)
    queue.submit({"task_id": "running-task", "agent_id": "alice", "npu_count": 1})
    before = queue.snapshot(None)["tasks"]
    sent = alice.message(to_bob, "这一轮之后可否释放？这不是 kill 命令。")
    incoming = poll(bob)["notifications"]
    assert len(incoming) == 1 and incoming[0]["message_id"] == sent["message"]["message_id"]
    assert incoming[0]["sender"] == "alice"
    assert incoming[0]["sender_session"] == alice.context["session"]["id"]
    answer = bob.reply(incoming[0]["reply_reference"], "本轮完成后处理。")
    reply = poll(alice)["notifications"][0]
    assert reply["thread_id"] == sent["message"]["thread_id"]
    assert reply["reply_to"] == sent["message"]["message_id"]
    assert reply["message_id"] == answer["message"]["message_id"]
    assert queue.snapshot(None)["tasks"] == before
    assert "notifications" not in bob.status()


def test_lost_send_reply_reuses_persisted_outbox_id(pair):
    backend, (alice, bob) = pair
    _, to_bob = addresses(pair)
    backend.fail_after_send = True
    with pytest.raises(TimeoutError):
        alice.message(to_bob, "请在方便时回复")
    sent = alice.message(to_bob, "请在方便时回复")
    messages = poll(bob)["notifications"]
    assert [row["message_id"] for row in messages] == [sent["message"]["message_id"]]
    requests = [row for row in backend.calls if row["action"] == "message"]
    assert requests[-2]["message_id"] == requests[-1]["message_id"]


def test_offline_mail_and_cursor_survive_local_coordinator_restart(pair):
    _, (alice, bob) = pair
    _, to_bob = addresses(pair)
    alice.message(to_bob, "离线期间留言")
    old = bob.coordinator
    bob._service = CoordinatorService(old.state_dir, pool=old.pool, backend=old.backend, sessions=bob.store)
    assert poll(bob)["notifications"][0]["text"] == "离线期间留言"
    join_poll(bob)
    bob._service = CoordinatorService(old.state_dir, pool=old.pool, backend=old.backend, sessions=bob.store)
    assert "notifications" not in poll(bob)
    with bob.store.transaction() as db:
        assert any(row["text"] == "离线期间留言" for row in bob.store.rows(db, "mail-message"))


def test_foreign_reply_and_unknown_host_are_rejected(pair):
    _, (alice, bob) = pair
    _, to_bob = addresses(pair)
    sent = alice.message(to_bob, "hello")
    with pytest.raises(CoordinationError, match="does not belong"):
        alice.reply({"host": to_bob["host"], "message_id": sent["message"]["message_id"]}, "spoof")
    with pytest.raises(ValueError, match="has not been used"):
        alice.message({**to_bob, "host": "unknown-host"}, "hello")
    with pytest.raises(ValueError, match="invalid coordination reference"):
        alice.message({**to_bob, "sender": "bob"}, "hello")


def test_blocked_host_never_blocks_status_or_starts_duplicate_worker(pair):
    backend, (alice, _) = pair
    backend.block = threading.Event()
    start = time.monotonic()
    alice.status()
    while not backend.calls and time.monotonic() - start < 1:
        time.sleep(.001)
    with ThreadPoolExecutor(4) as workers:
        list(workers.map(lambda _: alice.status(), range(8)))
    assert time.monotonic() - start < 1
    assert len(backend.calls) == 1
    backend.block.set()
    join_poll(alice)
    alice.status()
    assert len(backend.calls) == 1  # TTL also applies after the worker completes.


def test_local_session_and_finish_do_not_start_daemon(tmp_path, monkeypatch):
    context = AgentSessions(tmp_path / "sessions").attach("codex", "local", str(tmp_path))
    client = TaskClient(context["context_file"], user="alice")
    def forbidden(*args, **kwargs):
        raise AssertionError("local tasks must not start a daemon for messages")
    monkeypatch.setattr("mindie_coordinator.service.ensure_daemon", forbidden)
    assert client.status()["session"]["state"] == "open"
    assert client.finish()["state"] == "finished"


def test_host_epoch_reset_does_not_skip_new_low_cursor_messages(tmp_path):
    host = NpuCoordinator(tmp_path)
    first = host.message_events({"user": "a", "session_id": "task-alice"})
    host.message_events({"user": "b", "session_id": "task-bob"})
    host.message({"user": "b", "session_id": "task-bob", "recipient": "a", "recipient_session": "task-alice",
                  "message_id": "new-message", "text": "new epoch"})
    fresh = host.message_events({"user": "a", "session_id": "task-alice", "after": 1000,
                                 "mailbox_epoch": "lost-old-epoch"})
    assert fresh["events"][0]["text"] == "new epoch"
    assert fresh["mailbox_epoch"] == first["mailbox_epoch"]


def test_status_delivers_text_without_presentation_truncation(pair):
    from mindie_coordinator.presentation import present
    _, (alice, bob) = pair
    _, to_bob = addresses(pair)
    text = "x" * 3500
    alice.message(to_bob, text)
    value = poll(bob)
    result = present({"invocation_id": "test-message-output", "summary": "status", "data": value},
                     bob.coordinator.state_dir)
    assert result["data"]["notifications"][0]["text"] == text


def test_custom_backend_without_message_capability_is_unchanged(pair):
    backend, (alice, _) = pair
    backend.supports_task_messages = False
    assert "notifications" not in alice.status()
    assert backend.calls == []


def test_shutdown_does_not_write_to_local_store_after_blocked_fetch(pair):
    backend, (alice, _) = pair
    backend.block = threading.Event()
    alice.status()
    with ThreadPoolExecutor(1) as worker:
        closing = worker.submit(alice.coordinator.close_messages)
        assert alice.coordinator._mail_stopped.wait(1)
        backend.block.set()
        closing.result(timeout=5)
    assert not alice.coordinator._mail_workers
    with alice.store.transaction() as db:
        assert alice.store.rows(db, "mailbox") == []


def test_normal_run_and_execution_status_deliver_without_an_inbox_call(tmp_path):
    # Real admission, placement, host queue and status; only remote execution
    # is simulated by the existing coordinator test backend.
    from test_coordinator import Backend, runtime_spec
    backend = Backend(tmp_path / "shared-host")
    backend.supports_task_messages = True
    clients = []
    try:
        for user in ("alice", "bob"):
            store = AgentSessions(tmp_path / user / "sessions")
            context = store.attach("codex", user + "-real-run", str(tmp_path))
            pool = RuntimePool(tmp_path / user / "coordinator", backend)
            pool.register("runtime-" + user, runtime_spec(1, user=user))
            clients.append(TaskClient(context["context_file"], pool=pool, user=user))
        alice, bob = clients
        first = alice.run("true", sources={}, resources={"npu_count": 1, "devices": [0]})
        join_poll(alice)
        second = bob.run("true", sources={}, resources={"npu_count": 1, "devices": [0]})
        join_poll(bob)
        waiting = bob.observe(second["execution_id"], refresh=False)
        target = next(peer["reference"] for peer in waiting["coordination_peers"]
                      if peer["reference"]["user"] == "alice")
        bob.message(target, "当前任务预计何时结束？")
        expire_poll(alice)
        alice.observe(first["execution_id"], refresh=False)
        join_poll(alice)
        assert alice.observe(first["execution_id"], refresh=False)["notifications"][0]["sender"] == "bob"
    finally:
        for client in clients:
            if client._service is not None:
                client.coordinator.close_messages()
