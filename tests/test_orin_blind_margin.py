"""104 margin override must retain the platform physics."""
from pathlib import Path
import yaml


def test_three_metre_margin_preserves_wheel_physics():
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / 'config/wheel.yaml').read_text())
    hardware = yaml.safe_load((root / 'config/wheel_orin.yaml').read_text())
    assert hardware.pop('start_blind_zone_margin_m') == 3.0
    for name in ('maximum_forward_speed_mps', 'maximum_reverse_speed_mps'):
        assert hardware['capability'][name] == 0.2
        hardware['capability'][name] = base['capability'][name]
    assert hardware == base
