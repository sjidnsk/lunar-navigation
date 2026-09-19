import numpy as np


def test_soft_objective_separates_transform_effect_from_learning():
    from lunar_drl_exploration.actor_probe import summarize_fit
    first = dict(update=0, expected_q=.1, soft_objective=.15, best_probability=.01)
    last = dict(update=300, expected_q=.8, soft_objective=.82, best_probability=.9)
    row = summarize_fit(.05, [first, last])
    assert np.isclose(row['immediate_transform_gain'], .10)
    assert np.isclose(row['learning_gain'], .67)
    assert row['final_best_probability'] == .9


def test_q_switch_changes_teacher_only_and_leaves_initial_teacher_owned():
    import torch
    from lunar_drl_exploration.actor_probe import switched_q
    q = torch.tensor([1., 2., 3.])
    changed = switched_q(q)
    assert changed.argmax() == q.argmin()
    torch.testing.assert_close(q, torch.tensor([1., 2., 3.]))
    assert changed.max() > q.max()
