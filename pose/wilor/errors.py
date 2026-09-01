"""Exceptions raised by the WiLoR adapter, kept explicit rather than caught
and papered over -- callers must handle asset/hardware gaps deliberately."""

from __future__ import annotations

from pathlib import Path


class WilorAssetMissingError(RuntimeError):
    """A required checkpoint or MANO asset file is not present on disk."""

    def __init__(self, missing_path: Path, guidance: str) -> None:
        self.missing_path = missing_path
        super().__init__(f"Missing required WiLoR asset: {missing_path}\n{guidance}")


class ManoAssetMissingError(WilorAssetMissingError):
    """MANO_RIGHT.pkl is absent. This is the expected/documented blocker in
    environments without manual MANO license acceptance -- see
    reports/pose/wilor/TASK-002-wilor-karsl-pilot.md."""

    def __init__(self, missing_path: Path) -> None:
        from .config import MANO_LICENSE_URL

        guidance = (
            "MANO model files are gated behind manual account creation and "
            f"license acceptance at {MANO_LICENSE_URL}. They cannot be "
            "fetched automatically. Create an account, download "
            "mano_v*_*.zip, and place MANO_RIGHT.pkl at this path."
        )
        super().__init__(missing_path, guidance)


class WilorDependencyError(RuntimeError):
    """A required third-party package (wilor, ultralytics, smplx, ...) is
    not importable in the active environment."""
