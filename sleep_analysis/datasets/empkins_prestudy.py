import json
import re
from pathlib import Path

import pandas as pd
from tpcp import Dataset


class EmpkinsPreStudy(Dataset):
    def create_index(self):
        with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
            path_dict = json.load(f)
            path = Path(path_dict["empkins_prestudy"])

        path_list = list(Path(path).glob("Vp_*"))
        subjects = sorted(re.findall("(\d{2})", str(path_list)))

        return pd.DataFrame(subjects, columns=["subj_id"])

    @property
    def edf_path(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["empkins_prestudy"])

            edf_path = list(path.joinpath("Vp_" + self.index["subj_id"][0] + "/edf").glob("*.edf"))[0]
            return edf_path

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def data(self):
        pass
