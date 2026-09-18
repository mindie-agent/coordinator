"""Bound package programs without passing their bodies as a Bash argument."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

COMMAND_BYTES = 96 * 1024
PROGRAM_BYTES = 8 * 1024 * 1024


def preparation_command(script: str, *, setup=(), cwd=None) -> str:
    if not setup and len(script.encode('utf-8')) <= COMMAND_BYTES:
        return script
    data = json.dumps({'setup': list(setup), 'script': script, 'cwd': cwd},
                      separators=(',', ':')).encode('utf-8')
    if len(data) > PROGRAM_BYTES:
        raise ValueError('preparation program exceeds 8 MiB')
    encoded = base64.b64encode(zlib.compress(data)).decode('ascii')
    program = """import base64, hashlib, json, subprocess, sys, tempfile, time, zlib
data = zlib.decompress(base64.b64decode(__MINDIE_ENCODED__))
if hashlib.sha256(data).hexdigest() != __MINDIE_DIGEST__:
    raise ValueError('preparation program digest differs')
request = json.loads(data)
def run(script, cwd=None):
    # A private file avoids Linux's per-argument limit and leaves stdin alone.
    with tempfile.TemporaryFile() as source:
        source.write(script.encode('utf-8'))
        source.seek(0)
        result = subprocess.run(['bash', '/dev/fd/' + str(source.fileno())],
                                pass_fds=(source.fileno(),), cwd=cwd)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
for step, script in request['setup']:
    started = time.monotonic()
    run(script)
    print('__MINDIE_PARITY_PROGRESS__=' + json.dumps({'phase': step,
          'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
run(request['script'], request['cwd'])
""".replace('__MINDIE_ENCODED__', repr(encoded)).replace('__MINDIE_DIGEST__', repr(hashlib.sha256(data).hexdigest()))
    command = "python3 - <<'MINDIE_PREPARATION_PROGRAM'\n" + program + '\nMINDIE_PREPARATION_PROGRAM\n'
    if len(command.encode('utf-8')) > COMMAND_BYTES:
        raise ValueError('compressed preparation command exceeds 96 KiB')
    return command


def preparation_command_fits(script: str) -> bool:
    try:
        preparation_command(script)
        return True
    except ValueError:
        return False
