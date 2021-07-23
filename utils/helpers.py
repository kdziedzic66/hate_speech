import os
from typing import Any, Dict, List

import git
import numpy as np


def read_text_file(filepath: str) -> List[str]:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def broadcast_list_to_type(lst: List[Any], tp: Any) -> List[Any]:
    return [tp(el) for el in lst]


def get_aws_credentials_from_env() -> Dict[str, str]:
    credentials = {
        "app_key": os.environ.get("AWS_ACCESS_KEY", None),
        "secret_key": os.environ.get("AWS_SECRET_KEY", None),
    }
    return credentials


def get_repo_root() -> str:
    repo = git.Repo(".", search_parent_directories=True)
    repo_root = repo.working_tree_dir
    return repo_root