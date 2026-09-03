"""Non-rendering control-layer contracts for the virtual-glove visualizer.

TASK-007B owns the keyboard, Core-28 mapping, exemplar catalog and playback
queue.  Rendering and geometry implementations intentionally live elsewhere.
"""

__all__ = ["catalog", "keyboard", "mapping", "queue"]
