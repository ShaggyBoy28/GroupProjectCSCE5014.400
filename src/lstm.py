from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

WINDOW_SIZE = 32
TOT_ACTION_CLASSES = 6

openpose_to_detectron_mapping = [
    0, 1, 28, 29, 26, 27, 32, 33, 30, 31, 8, 9, 2, 3, 10, 11, 4, 5,
    12, 13, 6, 7, 20, 21, 14, 15, 22, 23, 16, 17, 24, 25, 18, 19,
]


class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.X_train_path = self.data_root / "X_train.txt"
        self.X_test_path = self.data_root / "X_test.txt"
        self.y_train_path = self.data_root / "Y_train.txt"
        self.y_test_path = self.data_root / "Y_test.txt"

    def convert_to_detectron_format(self, row):
        row = row.split(',')
        temp = row[:2] + row[4:]
        return [temp[i] for i in openpose_to_detectron_mapping]

    def load_X(self, X_path):
        with open(X_path, 'r') as file:
            X = np.array(
                [self.convert_to_detectron_format(row) for row in file],
                dtype=np.float32,
            )
        if len(X) % WINDOW_SIZE:
            raise ValueError(f"{X_path} must contain a multiple of {WINDOW_SIZE} frames")
        blocks = len(X) // WINDOW_SIZE
        if blocks == 0:
            raise ValueError(f"{X_path} contains no complete pose windows")
        return np.array(np.split(X, blocks))

    def load_y(self, y_path):
        with open(y_path, 'r') as file:
            y = np.array(
                [row.replace('  ', ' ').strip().split(' ') for row in file],
                dtype=np.int32,
            )
        return y - 1

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X_train = self.load_X(self.X_train_path)
        X_test = self.load_X(self.X_test_path)
        y_train = self.load_y(self.y_train_path)
        y_test = self.load_y(self.y_test_path)
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError("Each pose window must have exactly one action label")
        self.train_dataset = PoseDataset(X_train, y_train)
        self.val_dataset = PoseDataset(X_test, y_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


class ActionClassificationLSTM(pl.LightningModule):
    """LSTM over 32-frame windows of 17 keypoints (34 values per frame).

    Ported to PyTorch Lightning 2.x. Hyperparameters stay compatible with the shipped checkpoint
    (input_features=34, hidden_dim=50, learning_rate=1e-4); the new ones have defaults that match it.

    num_classes:  number of activities (6 for the shipped model; more after extending the dataset)
    class_names:  activity names in output order, saved in the checkpoint so the app can read them
    lr_monitor:   metric that drives the learning-rate scheduler. The original training used "val_loss";
                  cross-validation uses "train_loss" so the held-out fold never influences training.
    """

    def __init__(self, input_features, hidden_dim, learning_rate=0.001, num_classes=TOT_ACTION_CLASSES,
                 class_names=None, lr_monitor="val_loss"):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])

    def _step(self, batch, stage):
        x, y = batch
        y = y.reshape(-1).long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        # on_epoch=True averages over the epoch (replaces the removed *_epoch_end hooks)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # halve the learning rate when the monitored loss stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-15)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1,
                                 "monitor": self.hparams.lr_monitor}}
