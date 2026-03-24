from typing import Dict, List


class IPAToARPAbetConverter:
    """
    Based on the conversion style used in the user's components.py file,
    but reduced to what is needed for PLASMA metric grouping.
    """

    def __init__(self):
        self.ipa_to_arpabet: Dict[str, str] = {
            # vowels
            "i": "IY", "iː": "IY", "ɪ": "IH", "e": "EY", "eɪ": "EY", "ɛ": "EH",
            "æ": "AE", "a": "AE", "ə": "AH", "ʌ": "AH", "ɜ": "ER", "ɝ": "ER", "ɚ": "ER",
            "u": "UW", "uː": "UW", "ʊ": "UH", "o": "OW", "oʊ": "OW", "ɔ": "AO",
            "ɑ": "AA", "ɒ": "AA", "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY",

            # stops
            "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "g": "G", "ɡ": "G", "ʔ": "T",

            # fricatives
            "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
            "ʃ": "SH", "ʒ": "ZH", "h": "HH",

            # affricates
            "tʃ": "CH", "t͡ʃ": "CH", "ʧ": "CH", "dʒ": "JH", "d͡ʒ": "JH", "ʤ": "JH",

            # nasals
            "m": "M", "n": "N", "ŋ": "NG",

            # liquids / glides
            "l": "L", "ɹ": "R", "r": "R", "ɾ": "R", "w": "W", "j": "Y",

            # misc
            "SIL": "SIL", "SP": "SIL", "sil": "SIL", "": "SIL",
            "ˈ": "", "ˌ": "", "ː": "", ".": "",
        }

        self.arpabet_inventory = {
            "P", "B", "T", "D", "K", "G",
            "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
            "CH", "JH",
            "M", "N", "NG",
            "L", "R", "W", "Y",
            "IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER", "AY", "AW", "OY",
            "SIL",
        }

        self.class_map = {
            "P": "stops", "B": "stops", "T": "stops", "D": "stops", "K": "stops", "G": "stops",
            "F": "fricatives", "V": "fricatives", "TH": "fricatives", "DH": "fricatives",
            "S": "fricatives", "Z": "fricatives", "SH": "fricatives", "ZH": "fricatives", "HH": "fricatives",
            "CH": "affricates", "JH": "affricates",
            "M": "nasals", "N": "nasals", "NG": "nasals",
            "L": "liquids_glides", "R": "liquids_glides", "W": "liquids_glides", "Y": "liquids_glides",
            "IY": "front_vowels", "IH": "front_vowels", "EY": "front_vowels", "EH": "front_vowels", "AE": "front_vowels",
            "AA": "back_central_vowels", "AO": "back_central_vowels", "OW": "back_central_vowels",
            "UH": "back_central_vowels", "UW": "back_central_vowels", "AH": "back_central_vowels", "ER": "back_central_vowels",
            "AY": "diphthongs", "AW": "diphthongs", "OY": "diphthongs",
            "SIL": "silence",
        }

    def convert(self, symbol: str) -> str:
        if not symbol:
            return "SIL"

        s = symbol.strip()
        s_no_marks = s.replace("ˈ", "").replace("ˌ", "").replace("ː", "").replace(".", "").replace("͡", "")

        if s in self.ipa_to_arpabet:
            result = self.ipa_to_arpabet[s]
            if result:
                return result

        if s_no_marks in self.ipa_to_arpabet:
            result = self.ipa_to_arpabet[s_no_marks]
            if result:
                return result

        upper = s.upper()
        if upper in self.arpabet_inventory:
            return upper

        return "SIL"

    def phoneme_class(self, symbol: str) -> str:
        arpabet = self.convert(symbol)
        return self.class_map.get(arpabet, "other")