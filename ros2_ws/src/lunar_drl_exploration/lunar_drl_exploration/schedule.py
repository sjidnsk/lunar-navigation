"""Learner-owned exact update credit with reservation before collector dispatch."""
from fractions import Fraction


class UpdateSchedule:
    def __init__(self, warmup=1024, ratio=.25, max_credit=32):
        if int(warmup) != warmup or warmup < 0 or ratio <= 0 or max_credit < 1:
            raise ValueError('invalid update schedule')
        self.warmup = int(warmup)
        self.ratio = Fraction(str(ratio))
        self.max_credit = Fraction(str(max_credit))
        self.transitions = self.updates = self.inflight = 0

    def _credit_at(self, transitions):
        return max(0, transitions - self.warmup) * self.ratio - self.updates

    @property
    def credit(self):
        return self._credit_at(self.transitions)

    @property
    def can_update(self):
        return self.credit >= 1

    def can_dispatch(self, inflight_count=None, count=1):
        """Includes the proposed dispatch; caller count cannot omit own reservations."""
        inflight = self.inflight if inflight_count is None else inflight_count
        if int(inflight) != inflight or inflight < self.inflight or int(count) != count or count < 1:
            raise ValueError('invalid inflight/count')
        return self._credit_at(self.transitions + inflight + count) <= self.max_credit

    def reserve_dispatch(self, count=1):
        if not self.can_dispatch(count=count):
            raise ValueError('update credit backpressure')
        self.inflight += count

    def cancel_dispatch(self, count=1):
        if int(count) != count or not 1 <= count <= self.inflight:
            raise ValueError('no matching dispatch reservation')
        self.inflight -= count

    def collected(self, *, reserved=False):
        """Call once AFTER replay admission, then acknowledge the finished message."""
        if reserved:
            if self.inflight < 1:
                raise ValueError('no matching dispatch reservation')
        elif not self.can_dispatch():
            raise ValueError('update credit backpressure')
        if reserved:
            self.inflight -= 1
        self.transitions += 1

    def updated(self):
        """Only after a complete successful synchronous SAC update."""
        if not self.can_update:
            raise ValueError('no update credit')
        self.updates += 1

    def state_dict(self):
        return dict(schema='update_credit_v1', warmup=self.warmup,
                    ratio=str(self.ratio), max_credit=str(self.max_credit),
                    transitions=self.transitions, updates=self.updates,
                    credit=str(self.credit), unfinished_reservations=self.inflight)

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema') != 'update_credit_v1':
            raise ValueError('update schedule schema mismatch')
        result = cls(state['warmup'], Fraction(state['ratio']), Fraction(state['max_credit']))
        for key in ('transitions', 'updates'):
            value = state[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError('invalid schedule counter')
            setattr(result, key, value)
        if not 0 <= result.credit <= result.max_credit or str(result.credit) != state['credit']:
            raise ValueError('inconsistent schedule credit')
        # Unfinished motion is deliberately not stitched into new episodes.
        return result
