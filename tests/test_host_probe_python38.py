"""Rendered host probes must define their functions on pre-PEP585 Python."""
import builtins
import sys

from mindie_coordinator.provision import host_ops


def test_host_probe_annotations_do_not_evaluate_builtin_generics(monkeypatch):
    script = host_ops.render_host_probe_script()
    code = script.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    definitions = code.split('candidates = list(image.get("candidates") or [])', 1)[0]

    class Python38List(list):
        @classmethod
        def __class_getitem__(cls, item):
            raise TypeError("'type' object is not subscriptable")

    monkeypatch.setattr(sys, 'argv', ['probe', '{}', '46000:46999', 'mindie-', ''])
    namespace = {'__builtins__': {**vars(builtins), 'list': Python38List}}
    exec(compile(definitions, '<remote-host-probe>', 'exec', dont_inherit=True), namespace)
    assert callable(namespace['run'])
    assert namespace['run'].__annotations__['cmd'] == 'list[str]'
