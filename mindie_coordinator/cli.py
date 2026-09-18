"""Console entry for the local MindIE coordinator."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    from mindie_coordinator._stdio import configure_stdio
    configure_stdio()
    parser = argparse.ArgumentParser(
        prog="mindie-coordinator",
        description="Local MindIE coordinator: persistent environment, task isolation and resource placement.",
    )
    sub = parser.add_subparsers(dest="command")
    daemon = sub.add_parser("daemon", help="Run the persistent coordinator for this user/state-dir")
    daemon.add_argument("--state-dir", default="")
    daemon.add_argument("--action", choices=("serve", "status", "restart-if-idle"), default="serve")
    task = sub.add_parser("task-server", help="Serve MindIE task and coordination tools over stdio MCP")
    task.add_argument("--describe", action="store_true",
                      help="print the capability declaration and tool list as JSON, then exit")
    provision = sub.add_parser(
        "provision",
        help="Prepare this user's container mindie-<user> from a named image/recipe",
    )
    provision.add_argument("--host", required=True)
    provision.add_argument("--image", required=True, help="local-latest, rc, main, stable, or a full image reference")
    provision.add_argument("--user")
    provision.add_argument("--host-user", default="root")
    provision.add_argument("--host-port", type=int, default=22)
    provision.add_argument("--ssh-port", type=int)
    provision.add_argument("--machine-type", choices=("A2", "A3", "A5", "310P"))
    provision.add_argument("--password-env")
    register = sub.add_parser(
        "runtime-register",
        help="Adopt a prepared work root in this user's container (root==cwd)",
    )
    register.add_argument("--runtime-id", required=True)
    register.add_argument("--user")
    register.add_argument("--host", required=True)
    register.add_argument("--ssh-port", type=int, required=True)
    register.add_argument("--root", required=True, help="Absolute prepared work root inside the container")
    register.add_argument("--python", required=True)
    register.add_argument("--ssh-user", default="root")
    register.add_argument("--host-port", type=int, default=22)
    register.add_argument("--host-user", default="root")
    register.add_argument("--service-ports", default="")
    register.add_argument("--state-dir", default="", help="Coordinator state directory")
    register.add_argument("--reuse-only", action="store_true", help="Verify a native artifact donor without making its work root available for execution")
    register.add_argument("--source", action="append", default=[], metavar="NAME=PATH", help="Local source worktree matching a native artifact donor; repeat for each repository")
    args = parser.parse_args(argv)
    if args.command == "daemon":
        from mindie_coordinator.service import main as daemon_main
        return daemon_main(["--state-dir", args.state_dir, "--action", args.action])
    if args.command == "task-server":
        from mindie_coordinator.task_server import main as task_main
        return task_main(["--describe"] if args.describe else [])
    if args.command == "provision":
        from mindie_coordinator.provision import provision_user_container
        row = provision_user_container(
            host=args.host, image=args.image, user=args.user,
            host_user=args.host_user, host_port=args.host_port,
            ssh_port=args.ssh_port, machine_type=args.machine_type,
            password_env=args.password_env,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "runtime-register":
        from pathlib import Path
        from mindie_coordinator.client_paths import client_path
        from mindie_coordinator.execution_sources import capture_sources
        from mindie_coordinator.service import ensure_daemon, require_native_owner
        from mindie_coordinator.state_paths import coordinator_state_dir
        from mindie_coordinator.task_client import coordinator_user

        state = Path(client_path(args.state_dir)).expanduser().resolve() if args.state_dir else coordinator_state_dir()
        require_native_owner(state)
        sources = {}
        for item in args.source:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in sources:
                parser.error("--source requires distinct NAME=PATH entries")
            sources[name] = path
        if args.reuse_only and not {"vllm", "vllm-ascend"}.issubset(sources):
            parser.error("--reuse-only requires --source vllm=PATH and --source vllm-ascend=PATH")
        if sources and not args.reuse_only:
            parser.error("--source is only supported with --reuse-only")
        ports = [int(item) for item in args.service_ports.split(",") if item.strip()]
        spec = {
            "user": coordinator_user(args.user),
            "python": args.python,
            "host_endpoint": {"host": args.host, "port": args.host_port, "user": args.host_user},
            "endpoint": {"host": args.host, "port": args.ssh_port, "user": args.ssh_user, "root": args.root},
            "service_ports": ports,
            "reuse_only": args.reuse_only,
        }
        if args.reuse_only:
            spec["source_snapshot"] = capture_sources(sources, state)
        coordinator = ensure_daemon(state)
        coordinator.timeout = 300
        row = coordinator.runtime_register(args.runtime_id, spec)
        print(json.dumps({"runtime_id": row["id"], "user": row["user"], "container_name": row["container_name"],
                          "python": row["python"], "endpoint": row["endpoint"], "state": row["state"],
                          "reuse_only": row["reuse_only"]},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
