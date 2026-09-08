from typing import Dict


class IPAToARPAbetConverter:
    """
    Convert supported IPA variants to the canonical
    ARPAbet inventory used for PLASMA class grouping.

    Unsupported phones are classified as OTHER rather
    than SIL so that unfamiliar model outputs are not
    incorrectly treated as silence.
    """

    def __init__(self):
        self.ipa_to_arpabet: Dict[str, str] = {
            # Front vowels
            "i": "IY",
            "iː": "IY",
            "ɪ": "IH",
            "e": "EY",
            "eː": "EY",
            "ej": "EY",
            "eɪ": "EY",
            "ɛ": "EH",
            "ɛː": "EH",
            "æ": "AE",
            "æː": "AE",
            "a": "AE",

            # Central vowels
            "ə": "AH",
            "əː": "AH",
            "ɘ": "AH",
            "ʌ": "AH",
            "ʌː": "AH",
            "ɜ": "ER",
            "ɜː": "ER",
            "ɝ": "ER",
            "ɚ": "ER",

            # Back vowels
            "u": "UW",
            "uː": "UW",
            "ʊ": "UH",
            "o": "OW",
            "oː": "OW",
            "ow": "OW",
            "oʊ": "OW",
            "ɔ": "AO",
            "ɔː": "AO",
            "ɑ": "AA",
            "ɑː": "AA",
            "ɒ": "AA",

            # Diphthongs
            "aɪ": "AY",
            "aj": "AY",
            "aʊ": "AW",
            "aw": "AW",
            "ɔɪ": "OY",
            "oj": "OY",

            # Stops
            "p": "P",
            "b": "B",
            "t": "T",
            "d": "D",
            "k": "K",
            "g": "G",
            "ɡ": "G",

            # Fricatives
            "f": "F",
            "v": "V",
            "θ": "TH",
            "ð": "DH",
            "s": "S",
            "z": "Z",
            "ʃ": "SH",
            "ʒ": "ZH",
            "h": "HH",

            # Affricates
            "tʃ": "CH",
            "t͡ʃ": "CH",
            "ʧ": "CH",
            "dʒ": "JH",
            "d͡ʒ": "JH",
            "ʤ": "JH",

            # Nasals
            "m": "M",
            "n": "N",
            "ŋ": "NG",

            # Liquids and glides
            "l": "L",
            "r": "R",
            "ɹ": "R",
            "ɾ": "R",
            "w": "W",
            "j": "Y",

            # Explicit silence
            "SIL": "SIL",
            "SP": "SIL",
            "sil": "SIL",
        }

        self.arpabet_inventory = {
            "P", "B", "T", "D", "K", "G",
            "F", "V", "TH", "DH", "S", "Z",
            "SH", "ZH", "HH",
            "CH", "JH",
            "M", "N", "NG",
            "L", "R", "W", "Y",
            "IY", "IH", "EY", "EH", "AE",
            "AA", "AO", "OW", "UH", "UW",
            "AH", "ER",
            "AY", "AW", "OY",
            "SIL",
        }

        self.class_map = {
            "P": "stops",
            "B": "stops",
            "T": "stops",
            "D": "stops",
            "K": "stops",
            "G": "stops",

            "F": "fricatives",
            "V": "fricatives",
            "TH": "fricatives",
            "DH": "fricatives",
            "S": "fricatives",
            "Z": "fricatives",
            "SH": "fricatives",
            "ZH": "fricatives",
            "HH": "fricatives",

            "CH": "affricates",
            "JH": "affricates",

            "M": "nasals",
            "N": "nasals",
            "NG": "nasals",

            "L": "liquids_glides",
            "R": "liquids_glides",
            "W": "liquids_glides",
            "Y": "liquids_glides",

            "IY": "front_vowels",
            "IH": "front_vowels",
            "EY": "front_vowels",
            "EH": "front_vowels",
            "AE": "front_vowels",

            "AA": "back_central_vowels",
            "AO": "back_central_vowels",
            "OW": "back_central_vowels",
            "UH": "back_central_vowels",
            "UW": "back_central_vowels",
            "AH": "back_central_vowels",
            "ER": "back_central_vowels",

            "AY": "diphthongs",
            "AW": "diphthongs",
            "OY": "diphthongs",

            "SIL": "silence",
        }

    def convert(self, symbol: str) -> str:
        if symbol is None:
            return "OTHER"

        s = symbol.strip()

        if not s:
            return "OTHER"

        if s in self.ipa_to_arpabet:
            return self.ipa_to_arpabet[s]

        s_no_marks = (
            s.replace("ˈ", "")
            .replace("ˌ", "")
            .replace("ː", "")
            .replace(".", "")
        )

        if s_no_marks in self.ipa_to_arpabet:
            return self.ipa_to_arpabet[
                s_no_marks
            ]

        upper = s.upper()

        if upper in self.arpabet_inventory:
            return upper

        return "OTHER"

    def phoneme_class(self, symbol: str) -> str:
        arpabet = self.convert(symbol)

        if arpabet == "OTHER":
            return "other"

        return self.class_map.get(
            arpabet,
            "other",
        )