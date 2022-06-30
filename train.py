import numpy as np
import os
from matplotlib import pyplot as plt
import csv
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pit_wrapper import PITLossWrapper
from sdr import PairwiseNegSDR
import glob

from model import ConvTasNet
from wsj0_mix import Wsj0mixDataset
import warnings

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class TrainingModule:
    def __init__(self, conf):
        self.save_path = conf["other"]["save_path"]
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(conf, os.path.join(self.save_path, "conf"))
        self.log_iter = conf["other"]["log_iter"]
        self.start_scheduler_epoch = conf["other"]["start_scheduler_epoch"]
        self.end_epoch = conf["training"]["epochs"]
        self.n_src = conf["data"]["n_src"]
        self.log = {
            "train_sdr": [],
            "train_iter": [],
            "val_sdr": [],
            "val_iter": [],
        }
        # set model & optmizer
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ConvTasNet(**conf["model"]).to(self.device)
        self.dp = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if self.dp:
            n_gpu = torch.cuda.device_count()
            self.model = torch.nn.DataParallel(self.model, list(range(n_gpu)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), **conf["optim"])

        train_set = Wsj0mixDataset(
            csv_dir=conf["data"]["train_dir"],
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            segment=conf["data"]["segment"],
        )
        val_set = Wsj0mixDataset(
            csv_dir=conf["data"]["val_dir"],
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            segment=conf["data"]["segment"],
        )
        self.train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=conf["training"]["train_batch_size"],
            num_workers=conf["training"]["train_num_workers"],
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_set,
            shuffle=False,
            batch_size=conf["training"]["val_batch_size"],
            num_workers=conf["training"]["val_num_workers"],
            pin_memory=True
        )
        self.loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **conf["schedular"]
        )
        self.clipping_conf = conf["clipping"]
        self.early_stopping = EarlyStopping(patience=10)

        if conf["other"]["load_model"]:
            with open(os.path.join(self.save_path, "log.csv")) as f:
                reader = csv.reader(f)
                for key in self.log.keys():
                    self.log[key] = [float(i) for i in next(reader)]
            self.n_iter = self.log["train_iter"][-1]
            self.start_epoch = int(self.n_iter / len(self.train_loader)) + 1
            sdr_list = []
            for model_name in glob.glob(os.path.join(*[self.save_path, "models", "model_*"])):
                _idx = model_name.rindex("_") + 1
                sdr_list.append(float(model_name[_idx:]))
            self.top5_sdr = min(sdr_list) if len(sdr_list) > 0 else 0.0
            self.model.load_state_dict(
                torch.load(os.path.join(self.save_path, "saved_model"))
            )
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.save_path, "saved_optimizer"))
            )
            if self.start_epoch > self.start_scheduler_epoch:
                for i in reversed(range(conf["schedular"]["patience"])):
                    self.scheduler.step(self.log["val_sdr"][-i])
            if self.start_epoch > self.early_stopping.patience:
                for i in reversed(range(1, self.early_stopping.patience+1)):
                    self.early_stopping(-self.log["val_sdr"][-i])
                    assert not self.early_stopping.early_stop
        else:
            self.start_epoch = 1
            self.top5_sdr = 0.0
            self.n_iter = 0
        self.lr = self.optimizer.param_groups[0]["lr"]

    def train_model(self):
        self.model.train()
        with tqdm(total=len(self.train_loader), unit="batch") as pbar:
            for epoch in range(self.start_epoch, self.end_epoch + 1):
                pbar.set_description(f"[Epoch: {epoch}/{self.end_epoch}]")
                for mix, src in self.train_loader:
                    self.n_iter += 1
                    self.training_step(mix, src)
                    if self.n_iter % self.log_iter == 0 or self.n_iter == 1:
                        # self.eval_model()
                        self.log_progress()
                    if len(self.log["val_sdr"]) > 0:
                        pbar.set_postfix({
                            "TrainSDR": "%.2f" % (self.train_loss),
                            "EvalSDR": "%.2f" % (self.log["val_sdr"][-1]),
                        })
                    else:
                        pbar.set_postfix({
                            "TrainSDR": "%.2f" % (self.train_loss),
                            # "EvalSDR": "%.2f" % (self.log["val_sdr"][-1]),
                        })
                    pbar.update(1)
                pbar.refresh()
                pbar.reset()
                self.eval_model()
                if epoch > self.start_scheduler_epoch:
                    self.scheduler.step(self.log["val_sdr"][-1])
                if self.lr != self.optimizer.param_groups[0]["lr"]:
                    self.lr = self.optimizer.param_groups[0]["lr"]
                    print(self.lr)
                self.early_stopping(-self.log["val_sdr"][-1])
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

    def training_step(self, mix, src):
        mix = mix.unsqueeze(1).to(self.device)
        src = src.to(self.device)
        est_src = self.model(mix)
        loss = self.loss_func(est_src, src)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clipping_conf)
        self.optimizer.step()
        self.train_loss = -loss.item()
        self.optimizer.zero_grad()


    def eval_model(self):
        # log progress validation result
        self.model.eval()
        eval_loss = []
        with torch.no_grad():
            for mix, src in self.val_loader:
                # valuate one step
                mix = mix.unsqueeze(1).to(self.device)
                src = src.to(self.device)
                est_src = self.model(mix)
                loss = -self.loss_func(est_src, src)
                eval_loss.append(loss.item())
        self.log["val_sdr"].append(np.mean(eval_loss))
        self.log["val_iter"].append(self.n_iter)
        if self.log["val_sdr"][-1] > self.top5_sdr:
            os.makedirs(os.path.join(self.save_path, "models"), exist_ok=True)
            if self.dp:
                torch.save(
                    self.model.to("cpu").module.state_dict(),
                    os.path.join(self.save_path, "models", "model_%.3f" % (self.log["val_sdr"][-1]))
                )
            else:
                torch.save(
                    self.model.to("cpu").state_dict(),
                    os.path.join(self.save_path, "models", "model_%.3f" % (self.log["val_sdr"][-1]))
                )
            self.model.to(self.device)
            model_list = glob.glob(os.path.join(self.save_path, "models","model_*"))
            if len(model_list) > 5:
                sdr_list = []
                for name in model_list:
                    idx = name.rindex("_") + 1
                    sdr = name[idx:]
                    sdr_list.append(float(sdr))
                idx = sdr_list.index(min(sdr_list))
                del sdr_list[idx]
                os.remove(model_list[idx])
                self.top5_sdr = min(sdr_list)
        self.model.train()

    def log_progress(self):
        # log train progress
        self.log["train_sdr"].append(self.train_loss)
        self.log["train_iter"].append(self.n_iter)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "saved_model"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "saved_optimizer"))
        # save log.csv
        with open(os.path.join(self.save_path, "log.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for key in self.log.keys():
                writer.writerow(self.log[key])
        # plot log
        plt.figure()
        plt.plot(self.log["train_iter"], self.log["train_sdr"], "tab:blue")
        plt.plot(self.log["val_iter"], self.log["val_sdr"], "tab:orange")
        plt.ylim(0, max(self.log["train_sdr"])+1)
        plt.savefig(os.path.join(self.save_path, "log.png"))
        plt.close()
        plt.clf()



if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    torch.is_anomaly_enabled () # False
    torch.autograd._profiler_enabled () # False
    conf = {
        "data": {
            "train_dir": "E:/Users/Takahashi/wsj0-mix/2speakers/wav8k/min/tr",
            "val_dir": "E:/Users/Takahashi/wsj0-mix/2speakers/wav8k/min/cv",
            "sample_rate": 8000,
            "n_src": 2,
            "segment": 4,
        },
        "model": {
            "n_src": 2,
            "in_chan": 512,
            "n_blocks": 8,
            "n_repeats": 3,
            "bn_chan": 128,
            "hid_chan": 512,
            "skip_chan": 128,
            "conv_kernel_size": 3,
            "norm_type": "gLN",
            "mask_act": "relu",
            "causal": False,
        },
        "optim": {
            "lr": 1e-3
        },
        "schedular": {
            "mode": "max",
            "factor": 0.5,
            "patience": 2,
            # "threshold": 0.0001
        },
        "clipping": {
            "max_norm": 5.0,
            "norm_type": 2
        },
        "training": {
            "num_workers": 3,
            "epochs": 200,
            "train_batch_size": 4,
            "val_batch_size": 16,
            "train_num_workers": 3,
            "val_num_workers": 1,
        },
        "other": {
            "log_iter": 1000,
            "start_scheduler_epoch": 10,
            "save_path": "./result"
        }
    }
    conf["other"]["load_model"] = os.path.exists(
        os.path.join(conf["other"]["save_path"], "log.csv")
    )
    tm = TrainingModule(conf)
    tm.train_model()


