import io
import pytest
from lunar_drl_exploration.terminal import TrainingTerminal, format_status, clip


def record():
    return dict(transitions=1234, new_transitions=12, updates=50, wall_s=62,
        transitions_per_s=.19, updates_per_s=.8, credit=1.5, inflight=8,
        actor_version=48, replay_count=1234, replay_bytes=1048576, save_in_s=1800,
        environments={0: dict(family='moon', extent_m=60, state='EXECUTING', steps=12,
            budget=512, reference_coverage=.5, new_area_m2=0, reason_code='GOAL_REACHED')})


def test_semantics_missing_data_and_executing_not_last_success():
    lines = '\n'.join(format_status(record(), environments=2, warmup=1024))
    assert '执行目标' in lines and '上次导航 到达目标' in lines
    assert '[######------] 50.0%' in lines
    assert '新增 0.0m²' in lines and '已知 —m²' in lines
    assert 'E1 待定' in lines and '[????????????]' in lines
    assert '持续训练' in lines and '本次平均' in lines
    assert '00:30:00' in lines


def test_pipe_has_no_escapes_and_throttles_but_forces_final():
    stream = io.StringIO()
    ui = TrainingTerminal(environments=1, warmup=1024, stream=stream)
    ui.render(record()); first = stream.getvalue()
    ui.render(record()); assert stream.getvalue() == first
    ui.render(record(), phase='已保存并停止', force=True)
    assert '\033' not in stream.getvalue()
    assert '已保存并停止' in stream.getvalue()


def test_tty_refresh_and_event_preserve_history(monkeypatch):
    class TTY(io.StringIO):
        def isatty(self): return True
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr('lunar_drl_exploration.terminal.shutil.get_terminal_size', lambda _: (100, 40))
    stream = TTY(); ui = TrainingTerminal(environments=1, warmup=1024, stream=stream)
    ui.render(record()); ui.render(record()); ui.event('保存完成')
    assert '\033[' in stream.getvalue() and ui.rows == 0
    assert stream.getvalue().endswith('保存完成\n')
    assert clip('月表abc', 5) == '月表a'


def test_new_scene_does_not_invent_coverage():
    r = record(); r['environments'] = {0: dict(state='RESETTING', family='cave')}
    r['transitions'] = 15
    text = '\n'.join(format_status(r, environments=1, warmup=1024))
    assert '经验预热' in text and '初始化新场景' in text
    assert '50.0%' not in text and '[????????????]' in text


def test_collision_takes_precedence_over_coverage_termination():
    r=record()
    r['environments'][0].update(terminated=True,exhausted=False,reason_code='COLLISION')
    text='\n'.join(format_status(r,environments=1,warmup=1024))
    assert '碰撞终止' in text and '覆盖达标' not in text


@pytest.mark.parametrize('success,exhausted,label',[(True,False,'覆盖达标'),
    (True,True,'覆盖达标'),(False,True,'耗尽未达标')])
def test_terminal_distinguishes_success_and_exhaustion(success,exhausted,label):
    r=record();r['environments'][0].update(completed=success,exhausted=exhausted,terminated=True)
    text='\n'.join(format_status(r,environments=1,warmup=1024))
    assert label in text
