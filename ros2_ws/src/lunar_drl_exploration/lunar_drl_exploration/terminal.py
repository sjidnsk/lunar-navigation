"""Human-facing training display; no ownership of training or metric state."""
import math
import os
import shutil
import sys
import time
import unicodedata


def number(value, digits=1):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def clip(text, columns):
    result, used = '', 0
    for char in text:
        width = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + width > columns:
            break
        result += char
        used += width
    return result


def format_status(record, *, environments, warmup, target=None, phase=None):
    total = record['transitions']
    phase = phase or ('经验预热' if total < warmup else '采集与学习')
    target_text = f" / {target:,}" if target is not None else '（持续训练）'
    lines = [f"DRL 探索训练 | {phase} | 已运行 {duration(record['wall_s'])}",
        f"本次采集 {record['new_transitions']:,}{target_text} | 累计 {total:,} | 学习更新 {record['updates']:,}"]
    if total < warmup:
        lines.append(f'预热 {total:,} / {warmup:,} 条转移；预热后开始学习')
    lines.extend([
        f"本次平均：采集 {number(record['transitions_per_s'], 2)} 转移/s | 学习 {number(record['updates_per_s'], 2)} 更新/s",
        f"已发布策略版本 {record['actor_version']} | 待执行更新额度 {number(record['credit'], 2)} | 在途决策 {record['inflight']}",
        f"回放 {record['replay_count']:,} 条 / {record['replay_bytes']/1024**2:.1f} MiB | 自动保存倒计时 {duration(record['save_in_s'])}",
        '覆盖条=场景覆盖率；决策=已完成/回合上限；新增与倍率取上次动作'])
    states = {'READY': '等待决策', 'EXECUTING': '执行目标', 'RESETTING': '初始化新场景',
              'INITIALIZING': '初始化', 'CLOSED': '已关闭'}
    reasons = {'GOAL_REACHED': '到达目标', 'NO_PATH': '无路径', 'TIMEOUT': '导航超时', 'CANCELED': '已取消'}
    for env in range(environments):
        status = record['environments'].get(env, record['environments'].get(str(env), {}))
        ratio = status.get('reference_coverage', status.get('initial_reference_coverage'))
        valid = isinstance(ratio, (float, int)) and math.isfinite(ratio)
        filled = min(12, max(0, int(ratio * 12))) if valid else 0
        bar = '[' + '#' * filled + '-' * (12-filled) + ']' if valid else '[????????????]'
        coverage = f'{ratio:5.1%}' if valid else '    —'
        family = {'moon': '月表', 'cave': '洞穴'}.get(status.get('family'), '待定')
        state = states.get(status.get('state'), status.get('state', '等待启动'))
        if status.get('terminated'): state = '探索耗尽'
        elif status.get('truncated'): state = '预算截断'
        reason = status.get('reason_code')
        reason = reasons.get(reason, reason or '—')
        lines.append(f"E{env} {family} {number(status.get('extent_m'), 0)}m | {state} | 决策 {status.get('steps', 0)}/{status.get('budget', '—')} | {bar} {coverage}")
        lines.append(f"   已知 {number(status.get('known_area_m2'))}m² | 新增 {number(status.get('new_area_m2'))}m² | 路程 {number(status.get('distance_m'))}m | 倍率 {number(status.get('actual_rtf'))}× | 上次导航 {reason}")
    return lines


class TrainingTerminal:
    def __init__(self, *, environments, warmup, target=None, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.options = dict(environments=environments, warmup=warmup, target=target)
        self.interactive = self.stream.isatty() and os.environ.get('TERM') != 'dumb'
        self.rows = 0
        self.last_print = -math.inf

    def clear(self):
        if self.rows:
            self.stream.write(f'\033[{self.rows}A\r\033[J')
            self.rows = 0

    def event(self, message):
        self.clear()
        self.stream.write(message + '\n')
        self.stream.flush()

    def render(self, record, *, phase=None, force=False):
        now = time.monotonic()
        if not self.interactive and not force and now - self.last_print < 30:
            return
        lines = format_status(record, **self.options, phase=phase)
        columns, height = shutil.get_terminal_size((120, 40))
        # Small terminals use scrolling snapshots: never move above the viewport.
        inplace = self.interactive and len(lines) + 1 < height
        if self.interactive and not inplace and not force and now - self.last_print < 30:
            return
        self.clear()
        if inplace:
            lines = [clip(line, max(1, columns-1)) for line in lines]
        self.stream.write('\n'.join(lines) + '\n')
        self.stream.flush()
        self.rows = len(lines) if inplace else 0
        self.last_print = now
