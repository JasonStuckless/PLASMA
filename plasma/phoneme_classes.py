from __future__ import annotations

from typing import Dict

from plasma.decoding import normalize_phoneme_token


PVP_CLASSES = (
    "stops",
    "fricatives",
    "affricates",
    "nasals",
    "liquids_glides",
    "front_vowels",
    "back_central_vowels",
    "diphthongs",
)


class IPAToARPAbetConverter:
    """
    Compatibility wrapper for PLASMA phoneme-class grouping.

    Class assignment follows the phoneme inventory defined in the paper rather
    than ARPAbet vowel-category conventions. Unsupported symbols are assigned
    to ``other`` and remain available to sequence-level metrics.
    """

    def __init__(self):
        self.class_map: Dict[str, str] = {}

        self._add("stops", "p", "b", "t", "d", "k", "g", "ɡ")
        self._add("fricatives", "f", "v", "s", "z", "ʃ", "ʒ", "θ", "ð", "h")
        self._add("affricates", "tʃ", "t͡ʃ", "ʧ", "dʒ", "d͡ʒ", "ʤ")
        self._add("nasals", "m", "n", "ŋ")
        self._add("liquids_glides", "l", "r", "ɹ", "w", "j")
        self._add("front_vowels", "i", "iː", "ɪ", "e", "eː", "ɛ", "ɛː", "æ", "æː")
        self._add(
            "back_central_vowels",
            "u", "uː", "ʊ", "o", "oː", "ɔ", "ɔː", "ʌ", "ʌː", "ə", "əː", "ɑ", "ɑː",
        )
        self._add("diphthongs", "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "aj", "aw", "ej", "ow", "oj")

        self.silence_symbols = {"SIL", "SP", "sil"}

    def _add(self, phoneme_class: str, *symbols: str) -> None:
        for symbol in symbols:
            canonical = normalize_phoneme_token(symbol)
            self.class_map[canonical] = phoneme_class

    def convert(self, symbol: str | None) -> str:
        """Return a canonical symbol or OTHER for unsupported inputs."""
        canonical = normalize_phoneme_token(symbol)
        if not canonical:
            return "OTHER"
        if canonical in self.silence_symbols:
            return "SIL"
        if canonical in self.class_map:
            return canonical
        return "OTHER"

    def phoneme_class(self, symbol: str | None) -> str:
        canonical = self.convert(symbol)
        if canonical == "SIL":
            return "silence"
        if canonical == "OTHER":
            return "other"
        return self.class_map[canonical]
