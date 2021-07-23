import re
from typing import List

from utils.constants import EMOTICON_PATTERN, USERNAME_PATTERN, NON_ALPHANUMERIC_PATTERN


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
        super(UsernameRemover, self).__init__(pattern=USERNAME_PATTERN)


class EmoticonRemover(BaseTextCleaner):
    def __init__(self):
        super(EmoticonRemover, self).__init__(pattern=EMOTICON_PATTERN, unicode=True)


class SpecialSignsRemover(BaseTextCleaner):
    def __init__(self):
        super(SpecialSignsRemover, self).__init__(pattern=NON_ALPHANUMERIC_PATTERN)


class SpecialSignsRemover(BaseTextCleaner):
    def __init__(self):


REGISTRY = {"EmoticonRemover": EmoticonRemover, "UsernameRemover": UsernameRemover, "SpecialSignsRemover": SpecialSignsRemover}


class TextCleaningComposer:
    def __init__(self, cleaner_names: List[str]):
        self.cleaners = [REGISTRY[name]() for name in cleaner_names]

    def clean(self, text: str) -> str:
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text
