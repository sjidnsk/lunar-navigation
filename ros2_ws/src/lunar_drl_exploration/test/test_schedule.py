"""Credit is conserved across warmup, outstanding work and restart."""
import pytest
from lunar_drl_exploration.schedule import UpdateSchedule


def test_warmup_has_no_debt_and_exact_fractional_updates():
    schedule = UpdateSchedule()
    for _ in range(1024): schedule.collected()
    assert schedule.credit == 0 and not schedule.can_update
    for _ in range(4): schedule.collected()
    assert schedule.credit == 1
    schedule.updated()
    assert schedule.credit == 0
    for _ in range(400):
        schedule.collected()
        if schedule.can_update: schedule.updated()
    assert schedule.updates == 101
    with pytest.raises(ValueError): schedule.updated()


def test_inflight_capacity_reserved_before_dispatch_and_never_clamped():
    schedule = UpdateSchedule(warmup=2, ratio=.25, max_credit=2)
    for _ in range(10):
        assert schedule.can_dispatch(schedule.inflight)
        schedule.reserve_dispatch()
    assert not schedule.can_dispatch(10)
    with pytest.raises(ValueError): schedule.reserve_dispatch()
    for _ in range(10): schedule.collected(reserved=True)
    assert schedule.credit == 2 and schedule.inflight == 0
    with pytest.raises(ValueError): schedule.collected()
    assert schedule.transitions == 10 and schedule.credit == 2
    schedule.updated()
    for _ in range(4): schedule.reserve_dispatch()
    for _ in range(4): schedule.collected(reserved=True)
    assert schedule.credit == 2 and schedule.transitions == 14
    schedule.updated(); schedule.updated()
    assert schedule.updates == 3 and schedule.credit == 0


def test_restart_keeps_fractional_credit_discards_only_unfinished_reservations():
    schedule = UpdateSchedule(warmup=0, ratio=.1, max_credit=2)
    for _ in range(13): schedule.collected()
    schedule.updated(); schedule.reserve_dispatch(3)
    restored = UpdateSchedule.from_state_dict(schedule.state_dict())
    assert float(restored.credit) == .3 and restored.inflight == 0
    for _ in range(7): restored.collected()
    assert restored.credit == 1
    restored.updated()
    assert restored.updates == 2
    restored.reserve_dispatch(); restored.cancel_dispatch()
    assert restored.inflight == 0
    with pytest.raises(ValueError): UpdateSchedule.from_state_dict(dict(schedule.state_dict(), updates=9))
