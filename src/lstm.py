import numpy as np
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

WINDOW_SIZE = 32  # 32 continuous frames


class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

openpose_to_detectron_mapping = [0, 1, 28, 29, 26, 27, 32, 33, 30, 31, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 20, 21, 14, 15, 22, 23, 16, 17, 24, 25, 18, 19]

class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.X_train_path = self.data_root + "X_train.txt"
        self.X_test_path = self.data_root + "X_test.txt"
        self.y_train_path = self.data_root + "Y_train.txt"
        self.y_test_path = self.data_root + "Y_test.txt"


    # Detectron2 produces only 17 key points while OpenPose produces 18 (or more) key points.
    def convert_to_detectron_format(self, row):
        row = row.split(',')
        # filtering out coordinate of neck joint from the training/validation set originally generated using OpenPose.
        temp = row[:2] + row[4:]
        # change to Detectron2 order of key points
        temp = [temp[i] for i in openpose_to_detectron_mapping]
        return temp

    def load_X(self, X_path):
        file = open(X_path, 'r')
        X = np.array(
            [elem for elem in [
                self.convert_to_detectron_format(row) for row in file
            ]],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X) / WINDOW_SIZE)
        X_ = np.array(np.split(X, blocks))
        return X_

    # Load the networks outputs
    def load_y(self, y_path):
        file = open(y_path, 'r')
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()
        # for 0-based indexing
        return y - 1

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X_train = self.load_X(self.X_train_path)
        X_test = self.load_X(self.X_test_path)
        y_train = self.load_y(self.y_train_path)
        y_test = self.load_y(self.y_test_path)
        self.train_dataset = PoseDataset(X_train, y_train)
        self.val_dataset = PoseDataset(X_test, y_test)

    def train_dataloader(self):
        # train loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        return val_loader

# Number of output classes of the shipped checkpoint (models/saved_model.ckpt).
TOT_ACTION_CLASSES = 6


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
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

    def _step(self, batch, stage):
        x, y = batch
        y = torch.squeeze(y, dim=-1).long() if y.dim() > 1 else y.long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = torchmetrics.functional.accuracy(
            logits.argmax(dim=1), y, task="multiclass", num_classes=self.hparams.num_classes)
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
