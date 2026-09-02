"""Признаки кадра: реестр экстракторов, кэш и сами экстракторы."""

from adlearn.classification.features.base import REGISTRY, Extractor, get, register
from adlearn.classification.features.color import ColorExtractor
from adlearn.classification.features.shortcut import ShortcutExtractor
from adlearn.classification.features.visual import VisualExtractor

__all__ = [
    "REGISTRY",
    "ColorExtractor",
    "Extractor",
    "ShortcutExtractor",
    "VisualExtractor",
    "get",
    "register",
]
