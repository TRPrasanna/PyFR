"""Small signal-handling helpers for restartable RL jobs."""

import signal


_requested = False
_signum = None


def _handle_signal(signum, frame):
    global _requested, _signum

    _requested = True
    _signum = signum


def install_signal_handlers(signals=('USR1', 'TERM')):
    """Install lightweight handlers which request a clean stop."""
    for signame in signals:
        signum = getattr(signal, f'SIG{signame}', None)
        if signum is not None:
            signal.signal(signum, _handle_signal)


def request_preemption(signum=None):
    global _requested, _signum

    _requested = True
    _signum = signum


def preemption_requested():
    return _requested


def preemption_signal():
    return _signum
