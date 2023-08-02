import torch
from pytorch_lightning.callbacks import ModelSummary
from asr_project.model.lightning_model import ASRModelLightening
from asr_project.data.lightning_data_module import ASRDataModule
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import os


def asr_predict(ckpt_dir):
    config = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
    model_path = os.path.join(ckpt_dir, 'best.ckpt')
    model = ASRModelLightening.load_from_checkpoint(model_path, config=config)
    OmegaConf.update(config, "data.dataloader.validation_batch_size", 200)
    data_module = ASRDataModule(config.data)
    data_module.setup("predict")
    for batch in data_module.val_dataloader():
        print(batch.keys())
        print(batch['inputs'].shape)




if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(__file__))
    asr_predict(f'{base_path}/ckpt/phonemes')