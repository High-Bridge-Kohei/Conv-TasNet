import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import soundfile as sf
import pandas as pd
import random

class Wsj0mixDataset(Dataset):
    """Dataset class for the wsj0-mix source separation dataset.
    Args:
        csv_dir (str): The path to the directory containing the json files.
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        n_src (int, optional): Number of sources in the training targets.
    References
        "Deep clustering: Discriminative embeddings for segmentation and
        separation", Hershey et al. 2015.
    """

    dataset_name = "wsj0-mix"

    def __init__(self, csv_dir, n_src=2, sample_rate=8000, segment=4.0):
        super().__init__()
        # Task setting
        self.csv_dir = csv_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.like_test = self.seg_len is None
        if not os.path.exists(os.path.join(csv_dir, "info.csv")):
            print("Create info.csv from json file.")
            self.create_csv_from_json()
        self.df = pd.read_csv(os.path.join(csv_dir, "info.csv"), index_col=0)
        if self.seg_len is not None:
            drop_df = self.df[self.df["length"] < self.seg_len]
            drop_utt = len(drop_df)
            orig_utt = len(self.df)
            drop_time = drop_df["length"].values.sum() / 3600 / sample_rate
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_time, orig_utt, self.seg_len
                )
            )
        else:
            self.seg_len = None


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # Read sources
        sources_list = []
        for i in range(self.n_src):
            s, _ = sf.read(
                row[f"source_{i + 1}_path"], dtype="float32", start=start, stop=stop
            )
            sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(
            row["mixture_path"], dtype="float32", start=start, stop=stop
        )
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        return mixture, sources

    def create_csv_from_json(self):
        mix_json = os.path.join(self.csv_dir, "mix.json")
        sources_json = [
            os.path.join(self.csv_dir, source + ".json") for source in [f"s{n+1}" for n in range(self.n_src)]
        ]
        data = {}
        with open(mix_json, "r") as f:
            mix_infos = np.array(json.load(f))
        data["mixture_path"] = mix_infos[:, 0]
        sources_infos = []
        for n, src_json in enumerate(sources_json):
            with open(src_json, "r") as f:
                sources_infos.append(np.array(json.load(f)))
            data[f"source_{n+1}_path"] = sources_infos[-1][:, 0]
        data["length"] = mix_infos[:, 1]
        df = pd.DataFrame(data=data, index=np.arange(mix_infos.shape[0]))
        df.to_csv(os.path.join(self.csv_dir, "info.csv"))

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "sep_clean"
        infos["licenses"] = [wsj0_license]
        return infos


wsj0_license = dict(
    title="CSR-I (WSJ0) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC93S6A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)

if __name__ == "__main__":
    """
    Test Wsj0mixDataset
    """
    csv_dir = "E:/Users/Takahashi/wsj0-mix/2speakers/wav8k/min/cv"
    dataset = Wsj0mixDataset(csv_dir, 2, 8000, 4)

    dataloader = DataLoader(dataset, 4, pin_memory=True, num_workers=1)
    for mix, src in dataloader:
        print(mix.size(), src.size())
        break