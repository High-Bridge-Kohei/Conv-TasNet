import numpy as np
import os
import torch
import pandas as pd
import soundfile as sf
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
# from asteroid.losses import pairwise_neg_sisdr, singlesrc_neg_sisdr

from sdr import pairwise_neg_sisdr
from pit_wrapper import PITLossWrapper
import glob
from model import ConvTasNet
from wsj0_mix import Wsj0mixDataset

def main(
    conf,
    model_name,
    test_dir="E:/Users/Takahashi/wsj0-mix/2speakers/wav8k/min/tt"
):
    os.makedirs(save_path, exist_ok=True)
    batch_size = 4
    dataset = Wsj0mixDataset(
        csv_dir=test_dir,
        n_src=conf["data"]["n_src"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    sdr_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    model = ConvTasNet(**conf["model"]).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    sdr_list = []
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as pbar:
            for mix, src in dataloader:
                mix = mix.unsqueeze(1).to(device)
                src = src.to(device)
                est_src = model(mix)
                sdr = - sdr_func(est_src, src, reduction=False)
                sdr_list.append(sdr.cpu().numpy().copy().reshape(-1, 1))
                pbar.set_postfix({
                    "SI-SDR": "{:.2f}".format(sdr_list[-1].mean()),
                })
                pbar.update(1)
    sdr_list = np.vstack(sdr_list).reshape(-1)
    df = pd.DataFrame({
        "sdr": sdr_list,
    })
    sdr = float(model_name[model_name.rindex("_") + 1:])
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    df.to_csv(os.path.join(save_path, "results", "result_%.3f.csv" % (sdr)))
    print(
        "SI-SDR = %.2f + %.2f" % (
            df["sdr"].values.mean(), df["sdr"].values.std()
        )
    )

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    device = "cuda:0"
    # device = "cpu"
    save_path = "./result"
    conf = torch.load(os.path.join(save_path, "conf"))

    sdr_list = []
    results = glob.glob(os.path.join(save_path, "results", "result_*"))
    models = glob.glob(os.path.join(save_path, "models", "model_*"))
    for model_name in reversed(models):
        eval_sdr = model_name[model_name.rindex("_") + 1:]
        result_path = os.path.join(save_path, "results", f"result_{eval_sdr}.csv")
        if not result_path in results:
            main(conf, model_name)
        sdr = pd.read_csv(result_path, index_col=0)["sdr"].values
        sdr_list.append(sdr.reshape(-1, 1))
    sdr_list = np.hstack(sdr_list)
    print("SDR = %.2f +- %.2f" % (sdr_list.mean(), sdr_list.std()))

