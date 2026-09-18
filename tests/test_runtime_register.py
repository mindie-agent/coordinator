from unittest.mock import MagicMock, Mock, patch

from mindie_coordinator.cli import main
from mindie_coordinator.service import CoordinatorClient, CoordinatorService


def test_runtime_registration_uses_daemon_and_preserves_explicit_spec(tmp_path, capsys):
    pool = MagicMock()
    pool.get.side_effect = ValueError("no saved session directories")
    pool.register.return_value = {"id": "donor", "state": "ready"}
    service = CoordinatorService(tmp_path / "coordinator", pool=pool)
    client = CoordinatorClient(tmp_path / "coordinator")
    spec = {"reuse_only": True, "source_snapshot": {"id": "fixed-descriptor"}}
    with patch.object(client, "call", side_effect=lambda op, **payload: service.handle({"op": op, **payload})["value"]):
        assert client.runtime_register("donor", spec) == pool.register.return_value
    pool.register.assert_called_once_with("donor", spec)


def test_existing_registration_cli_routes_through_daemon_without_local_pool(tmp_path, capsys):
    client = Mock()
    client.runtime_register.return_value = {"id": "donor", "user": "alice", "container_name": "mindie-alice",
                                           "python": "/venv/bin/python", "endpoint": {}, "state": "ready", "reuse_only": False}
    with patch("mindie_coordinator.service.ensure_daemon", return_value=client) as connect, \
         patch("mindie_coordinator.ready_runtime.RuntimePool", side_effect=AssertionError("client must not open a local pool")):
        assert main(["runtime-register", "--runtime-id", "donor", "--user", "alice", "--host", "192.0.2.1",
                     "--ssh-port", "46001", "--root", "/donor", "--python", "/venv/bin/python", "--state-dir", str(tmp_path)]) == 0
    assert connect.call_args.args[0] == tmp_path.resolve()
    assert client.runtime_register.call_args.args[0] == "donor"
    assert client.runtime_register.call_args.args[1]["reuse_only"] is False
