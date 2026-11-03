"""Driver 25.7 A5 PCIe rows retain measured HBM and process ownership."""
import pytest

from mindie_coordinator.host import mindie_npu_coordination as host


TABLE = """| NPU ID | Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
|        |      | Bus-Id | NPU Util(%) Memory-Usage(MB) HBM-Usage(MB) |
| 0 | Ascend950PR | OK | 224.4 65 0 / 0 |
|   |             | 0000:01:00.0 | 0 0 / 0 5155 / 131072 |
| 1 | Ascend950PR | OK | 210.1 64 0 / 0 |
|   |             | 0000:11:00.0 | 0 0 / 0 5154 / 131072 |
| NPU ID | Process id | Process name | Process memory(MB) | Process id in container |
| No running processes found in NPU 0 |
{process}
"""


@pytest.mark.parametrize('busy', [False, True])
def test_a5_pcie_rows_preserve_hbm_and_process_identity(busy):
    process = '| 1 | 12345 | python3 | 2048 | 42 |' if busy else '| No running processes found in NPU 1 |'
    text = TABLE.format(process=process)
    result = host.parse_npu_smi_info(text)
    assert result['status'] == 'ok', result
    assert result['devices'] == [0, 1]
    assert result['free'] == ([0] if busy else [0, 1])
    assert host.parse_npu_smi_hbm(text) == {
        0: {'used_mb': 5155, 'total_mb': 131072},
        1: {'used_mb': 5154, 'total_mb': 131072},
    }
    if busy:
        assert result['busy']['1'][0]['pid'] == 12345
        assert result['busy']['1'][0]['memory_mb'] == 2048


def test_truncated_a5_pcie_table_never_grants_unobserved_devices():
    text = TABLE.format(process='| No running processes found in NPU 1 |')
    text = text.replace('|   |             | 0000:11:00.0 | 0 0 / 0 5154 / 131072 |\n', '')
    result = host.parse_npu_smi_info(text)
    assert result['status'] == 'failed'
    assert result['free'] == []
