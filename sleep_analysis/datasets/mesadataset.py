import json
import re
from pathlib import Path

import pandas as pd
from tpcp import Dataset

from sleep_analysis.datasets.helper import build_base_path_processed_mesa


class MesaDataset(Dataset):
    """
    Dataset class for the MESA dataset created according to the tpcp framework (https://github.com/mad-lab-fau/tpcp)
    """

    def create_index(self):
        path = build_base_path_processed_mesa()

        path = path.joinpath("features_full_combined").resolve()
        path_list = list(Path(path).glob("*.csv"))
        subj_id = re.findall("(\d{4})", str(path_list))
        return pd.DataFrame(subj_id, columns=["subj_id"])

    @property
    def actigraph_data(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[["activity"]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def ground_truth(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[
                ["sleep", "5stage", "4stage", "3stage"]
            ]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def feature_table(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("features_full_combined").resolve()
            return pd.read_csv(path.joinpath("features_combined" + self.index["subj_id"][0] + ".csv"), index_col=0)

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def time(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[["linetime"]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def information(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path"])
            df = pd.read_csv(path.joinpath("datasets/mesa-sleep-dataset-0.5.0.csv"), index_col="mesaid").fillna(0)
            df.index = df.index.astype(str).str.zfill(4)
            return df[
                [
                    "race1c",
                    "gender1",
                    "overall5",
                    "whiirs5c",
                    "slpapnea5",
                    "insmnia5",
                    "rstlesslgs5",
                    "sleepage5c",
                    "ahi_a0h4",
                    "extrahrs5",
                ]
            ].loc[self.index["subj_id"][0]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def tst(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path"])
            df = pd.read_csv(
                path.joinpath("datasets/mesa-sleep-harmonized-dataset-0.5.0.csv"), index_col="mesaid"
            ).fillna(0)
            df.index = df.index.astype(str).str.zfill(4)
            return df[["nsrr_ttldursp_f1"]].loc[self.index["subj_id"][0]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def edf_path(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path_edf"])
            filename = re.findall("mesa-sleep-" + self.index["subj_id"][0] + ".edf", str(list(path.glob("*.edf"))))[0]
            return path.joinpath(filename)

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )
