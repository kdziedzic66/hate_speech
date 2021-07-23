import re
from typing import List

from utils.constants import EMOTICON_PATTERN, USERNAME


class BaseTextCleaner:
    def __init__(self, pattern: str, unicode: bool = False):
        flags = 0 if not unicode else re.UNICODE
        self.pattern = re.compile(pattern, flags=flags)

    def clean(self, text: str) -> str:
        cleaned_text = self.pattern.sub("", text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text


class UsernameRemover(BaseTextCleaner):
    def __init__(self):
        super(UsernameRemover, self).__init__(pattern=USERNAME)


class EmoticonRemover(BaseTextCleaner):
    def __init__(self):
        super(EmoticonRemover, self).__init__(pattern=EMOTICON_PATTERN)


REGISTRY = {"EmoticonRemover": EmoticonRemover, "UsernameRemover": UsernameRemover}


class TextCleaningComposer:
    def __init__(self, cleaner_names: List[str]):
        self.cleaners = [REGISTRY[name]() for name in cleaner_names]

    def clean(self, text: str) -> str:
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text
