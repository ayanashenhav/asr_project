import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.lightning_data_module import ASRDataModule
from ..model.lightning_model import ASRModelLightening
from ..utils.utils import get_config

DEFAULT_CONFIG_PATH = os.path.expanduser('~/python3/languages/hear/audiolm/audiolm/default_configs')


def main_train(full_path):
    config = get_config(full_path, DEFAULT_CONFIG_PATH)
    model = ASRModelLightening(config)
    data_module = ASRDataModule(config, model.get_input_names())

    # TODO: Change to W&B
    logger = TensorBoardLogger(save_dir=os.path.join(full_path, 'tb_logs'), name='logs/')

    checkpoint_callback = ModelCheckpoint(monitor='val/loss', dirpath=full_path, save_last=True,
                                          save_top_k=-1, every_n_epochs=1)

    trainer = pl.Trainer(default_root_dir=full_path,
                         logger=logger,
                         # overfit_batches=1,
                         log_every_n_steps=200,
                         callbacks=[checkpoint_callback, ModelSummary(max_depth=3)],
                         # track_grad_norm=2,
                         **config['train.trainer'],
                         )
    trainer.fit(model=model, ckpt_path="last", datamodule=data_module)
