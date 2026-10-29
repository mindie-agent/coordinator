"""Consume the host's compile share at every build environment boundary.

[Role] Normalize compiler parallelism and generate the shell environment clamp.
[Invariants] All three build knobs share the lowest explicit positive limit and grant.
[Timeouts & Locks] Pure arithmetic/text generation; no waits or locks.
[Side-Effect Policy] Returns environment values and shell text only.
"""
BUILD_JOB_VARIABLES = ('MAX_JOBS', 'MINDIE_BUILD_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL')


def compile_environment(budget, requested=None):
    if not budget or budget.get('status') != 'granted':
        return dict(requested or {})
    cap = budget['jobs_per_slot']
    if type(cap) is not int or cap < 1:
        raise ValueError('compile grant has no worker capacity')
    jobs = cap
    for key in BUILD_JOB_VARIABLES:
        value = (requested or {}).get(key)
        if value is not None:
            if not str(value).isdecimal() or int(value) < 1:
                raise ValueError(f'{key} must be a positive integer')
            jobs = min(jobs, int(value))
    return {**(requested or {}), **{key: str(jobs) for key in BUILD_JOB_VARIABLES},
            'MINDIE_COMPILE_JOBS_LIMIT': str(jobs)}


# Source capture and remote image setup can both supply build variables after
# process launch. Reapply the same lease cap after those inputs, before compile.
COMPILE_ENV_CLAMP = '''if [ -n "${MINDIE_COMPILE_JOBS_LIMIT:-}" ]; then
  mindie_compile_jobs="$MINDIE_COMPILE_JOBS_LIMIT"
  for mindie_compile_key in MAX_JOBS MINDIE_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL; do
    mindie_compile_value="${!mindie_compile_key:-}"
    if [ -n "$mindie_compile_value" ]; then
      case "$mindie_compile_value" in *[!0-9]*) echo "invalid $mindie_compile_key" >&2; exit 2;; esac
      if [ "$mindie_compile_value" -lt 1 ]; then echo "invalid $mindie_compile_key" >&2; exit 2; fi
      if [ "$mindie_compile_value" -lt "$mindie_compile_jobs" ]; then mindie_compile_jobs="$mindie_compile_value"; fi
    fi
  done
  export MAX_JOBS="$mindie_compile_jobs" MINDIE_BUILD_JOBS="$mindie_compile_jobs" CMAKE_BUILD_PARALLEL_LEVEL="$mindie_compile_jobs"
fi'''
