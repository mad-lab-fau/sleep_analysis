import json
from pathlib import Path


def get_db_path():
    with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
        path_dict = json.load(f)
        db_path = Path(path_dict["database_storage"])

        return str(db_path)
