import re
from Utils.phonemize.cotlet_phon import phonemize
from Utils.phonemize.cotlet_phon_dir_backend import latn_phonemize

# make sure you have correct spacing when using a mixture of japanese and romaji otherwise it goes into alphabet reading mode.

def is_japanese(text):

    japanese_ranges = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # Kanji
    ]
    
    for char in text:
        char_code = ord(char)
        for start, end in japanese_ranges:
            if start <= char_code <= end:
                return True
    return False

def has_only_japanese(text):
    # Remove spaces and check if all remaining characters are Japanese
    text_no_spaces = ''.join(char for char in text if not char.isspace())
    return all(is_japanese(char) for char in text_no_spaces)

def has_only_romaji(text):
    # Remove spaces and check if all remaining characters are ASCII
    text_no_spaces = ''.join(char for char in text if not char.isspace())
    return all(ord(char) < 128 for char in text_no_spaces)

def mixed_phonemize(text):
    # Split text into words while preserving spaces
    words = re.findall(r'\S+|\s+', text)
    result = []
    
    for word in words:
        if word.isspace():
            result.append(word)
            continue
            
        if is_japanese(word):
            result.append(phonemize(word))
        else:
            result.append(latn_phonemize(word))
    
    return ''.join(result)

def smart_phonemize(text):
    if has_only_japanese(text):
        return phonemize(text)
    elif has_only_romaji(text):
        return latn_phonemize(text)
    else:
        return mixed_phonemize(text)