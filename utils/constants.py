POLBERT_PRETRAINED_ID = "dkleczek/bert-base-polish-cased-v1"
EMOTICON_PATTERN = "[" u"\U0001F600-\U0001F64F" "]+"
NON_ALPHANUMERIC_PATTERN = '\W+'
USERNAME_PATTERN = "@[\w]+"
NUM_HATE_CLASSES = 3


CLASS_NAME_MAPPING = {0: "non-harmfull", 1: "cyberbulling", 2: "hate-speech"}
