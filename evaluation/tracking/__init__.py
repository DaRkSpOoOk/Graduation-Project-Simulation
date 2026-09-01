"""Neutral validation of derived hand-tracking output against an independent
human benchmark.

This package only *reads* frozen tracker output and frozen annotations. It
contains no tracking logic, no thresholds that feed back into the tracker,
and it never modifies either frozen input.
"""
