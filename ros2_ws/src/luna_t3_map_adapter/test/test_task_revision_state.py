from luna_t3_map_adapter.task_revision_state import TaskRevisionState


def test_only_active_task_and_new_map_revision_request_publication() -> None:
    state = TaskRevisionState()

    assert not state.activate("mission", 1, 1, (0.0, 0.0, 100.0, 100.0))
    assert state.observe_map_revision(7)
    assert not state.observe_map_revision(7)
    assert state.observe_map_revision(8)


def test_pausing_or_revising_task_invalidates_pending_publication() -> None:
    state = TaskRevisionState()
    state.activate("mission", 1, 1, (0.0, 0.0, 100.0, 100.0))
    assert state.observe_map_revision(7)
    assert not state.activate("mission", 2, 2, (0.0, 0.0, 100.0, 100.0))
    assert not state.pause()
    assert not state.observe_map_revision(8)
