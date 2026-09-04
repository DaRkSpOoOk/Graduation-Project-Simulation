"""Authoritative Core-28 character-to-SignID resolution."""

from .core28 import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_LABELS_PATH,
    CharacterResolution,
    Core28Label,
    Core28Mapping,
    Core28Resolver,
    UNSUPPORTED_CORE28_SEQUENCES,
    UnsupportedCharacterError,
    resolve_character,
    unsupported_sequence_at,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LABELS_PATH",
    "CharacterResolution",
    "Core28Label",
    "Core28Mapping",
    "Core28Resolver",
    "UNSUPPORTED_CORE28_SEQUENCES",
    "UnsupportedCharacterError",
    "resolve_character",
    "unsupported_sequence_at",
]
