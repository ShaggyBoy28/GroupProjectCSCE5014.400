import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from .lstm import ActionClassificationLSTM, PoseDataModule, openpose_to_detectron_mapping


def do_training_validation(argv):
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_root", required=True, help="directory containing X_train.txt, Y_train.txt, X_test.txt, and Y_test.txt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    args = parser.parse_args(argv)

    pl.seed_everything(21, workers=True)
    hidden_dim = 50
    model = ActionClassificationLSTM(
        len(openpose_to_detectron_mapping), hidden_dim, learning_rate=args.learning_rate
    )
    data_module = PoseDataModule(data_root=args.data_root, batch_size=args.batch_size)

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_callback = ModelCheckpoint(
        dirpath=project_root / "models",
        filename="saved_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=callbacks,
        default_root_dir=project_root,
    )
    trainer.fit(model, datamodule=data_module)
    return model


if __name__ == '__main__':
    do_training_validation(sys.argv[1:])
