"""
log.py — Centralised verbosity control.

All src/ modules import `vprint` from here instead of using print() directly.
Grid search sets VERBOSE = False before each run to silence all output.
"""

VERBOSE: bool = True


def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


def set_verbose(verbose: bool) -> None:
    global VERBOSE
    VERBOSE = verbose
