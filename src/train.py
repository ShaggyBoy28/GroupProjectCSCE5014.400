"""Original training entry point, ported to PyTorch Lightning 2.x.

    python -m src.train -d <folder with X_train.txt, Y_train.txt, X_test.txt, Y_test.txt>/

Trains on X_train/Y_train, validates on X_test/Y_test, and keeps the checkpoint with the lowest
validation loss, as the original training did. For cross-validation and quality/scale reports use
scripts/train_model.py instead.

Fixes against the 1.3-era version: Trainer.add_argparse_args/from_argparse_args, gpus= and
progress_bar_refresh_rate= no longer exist, and the model was built with WINDOW_SIZE (32) inputs
instead of 34 (17 keypoints x 2).
"""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from .lstm import ActionClassificationLSTM, PoseDataModule

INPUT_FEATURES = 34  # 17 Detectron2 keypoints x (x, y)


def do_training_validation(argv=None):
    parser = argparse.ArgumentParser(description="Train the activity LSTM on the original split")
    parser.add_argument("-d", "--data_root", required=True, help="folder with the dataset .txt files (end with /)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=50)
    args = parser.parse_args(argv)

    pl.seed_everything(21)
    data_root = args.data_root if args.data_root.endswith("/") else args.data_root + "/"
    model = ActionClassificationLSTM(INPUT_FEATURES, args.hidden_dim, learning_rate=args.learning_rate)
    data_module = PoseDataModule(data_root=data_root, batch_size=args.batch_size)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        deterministic=True,
        callbacks=[EarlyStopping(monitor="train_loss", patience=15),
                   ModelCheckpoint(save_top_k=1, monitor="val_loss"),
                   LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, data_module)
    return model


if __name__ == "__main__":
    do_training_validation()
