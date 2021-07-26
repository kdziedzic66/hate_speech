import json
import os
from typing import Any, Dict, List

import git
import wget


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


def save_json(data: Any, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Any:
    with open(filepath) as f:
        data = json.load(f)
    return data


def download_model_from_s3(model_name: str):
    weights_url = (
        f"https://hatespeechml.s3.eu-central-1.amazonaws.com/{model_name}/weights.pt"
    )
    config_url = f"https://hatespeechml.s3.eu-central-1.amazonaws.com/{model_name}/pipeline_config.json"
    output_dir = os.path.join(get_repo_root(), "trained_models", model_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    _ = wget.download(weights_url, out=output_dir)
    _ = wget.download(config_url, out=output_dir)
