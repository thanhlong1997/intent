from pyvi import ViTokenizer
import re
pt= re.compile(r"_")
def segmentation(text):
    return ViTokenizer.tokenize(text)

def split_words(text):
    text = segmentation(text)
    try:
        return [x.strip("0123456789%@$.,=+-!;/()*\"&^:#|\n\t\'").lower() for x in text.split()]
    except TypeError:
        return []

def get_words_feature(text):
    split_word = split_words(text)
    return [word for word in split_word if word.encode('utf-8')]
