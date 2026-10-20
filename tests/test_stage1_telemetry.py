"""Real A2 capture verifies exact-window sampling and process-schema handling."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parent


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'live' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT = load_helper('project_stage1_telemetry')
PROCESSES = load_helper('stage1_process_stream')


@pytest.fixture
def capture():
    return json.loads((ROOT / 'fixtures/a2-stage1-high-rate-window.json').read_text())


def summarize(capture, rows):
    return PROJECT.getter_window(rows, 'current_freq',
        capture['window']['start_ts'] * 1e9, capture['window']['end_ts'] * 1e9,
        capture['interval_ns'])


def test_complete_real_window_excludes_neighbors_and_counts_every_scheduled_read(capture):
    result = summarize(capture, capture['samples'])
    assert result['coverage_ratio'] == 1.0
    assert len(result['reads']) == 7
    assert result['missing_sample_indices'] == []
    assert all(read['value'] == 1650 for read in result['reads'])
    assert result['neighbor_before']['sample_index'] not in result['observed_sample_indices']
    assert result['neighbor_after']['sample_index'] not in result['observed_sample_indices']


@pytest.mark.parametrize('missing_row', [True, False])
def test_missing_or_failed_real_sample_cannot_be_reported_as_full_coverage(capture, missing_row):
    rows = capture['samples']
    missing_index = rows[4]['sample_index']
    if missing_row:
        rows.pop(4)
    else:
        getter = rows[4]['cards']['0']['getters']['current_freq']
        getter.update(ret=-1, error='observed getter failed', value=None)
    result = summarize(capture, rows)
    assert result['coverage_ratio'] == pytest.approx(6 / 7)
    assert result['missing_sample_indices'] == [missing_index]


def test_real_a2_busy_process_memory_schema_and_unknown_output(capture):
    parsed = PROCESSES.parse_process_memory(capture['proc_mem_stdout'])
    assert parsed == [{'pid': 3466550, 'name': 'python3', 'memory_mb': 130}]
    assert PROCESSES.parse_process_memory('\tNo process in device.\n') == []
    with pytest.raises(ValueError):
        PROCESSES.parse_process_memory(capture['proc_mem_stdout'] + 'query failed\n')


def test_actual_vllm_worker_is_owned_through_its_stable_marked_parent():
    capture = json.loads((ROOT / "fixtures/a3-vllm-process-ancestry.json").read_text())
    chain = capture["process"]["ancestry"]
    assert chain[0]["job_marker_matches"] is False
    assert PROCESSES.ownership_from_chain(chain, complete=True) is True
    # Losing that parent's identity makes ownership unknown, never foreign.
    marked = next(item for item in chain if item["job_marker_matches"])
    marked["identity_stable"] = False
    assert PROCESSES.ownership_from_chain(chain, complete=True) is None


def test_actual_unmarked_worker_without_ancestor_evidence_stays_unknown():
    capture = json.loads((ROOT / "fixtures/a3-vllm-process-ancestry.json").read_text())
    chain = capture["process"]["ancestry"][:1]
    assert PROCESSES.ownership_from_chain(chain, complete=False) is None
