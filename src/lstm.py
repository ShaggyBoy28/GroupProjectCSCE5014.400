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
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)

    def forward(self, x):
        _, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1).long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        accuracy = (y_pred.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1).long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        accuracy = (y_pred.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-15,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
