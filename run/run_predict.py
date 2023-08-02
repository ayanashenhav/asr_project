import torch
from pytorch_lightning.callbacks import ModelSummary
from asr_project.model.lightning_model import ASRModelLightening
from asr_project.data.lightning_data_module import ASRDataModule
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import os


def asr_predict(ckpt_dir):
    config = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

    OmegaConf.update(config, "data.dataloader.validation_batch_size", 200)
    data_module = ASRDataModule(config)
    data_module.setup("predict")
    print('validation')
    for batch in data_module.val_dataloader():
        break

    for model_path in os.listdir(ckpt_dir):
        if not model_path.endswith('.ckpt'):
            continue
        print(model_path)
        model_path = os.path.join(ckpt_dir, model_path)
        model = ASRModelLightening.load_from_checkpoint(model_path, config=config)

        batch['input'] = batch['input'].to(model.device)
        batch['input_lengths'] = batch['input_lengths'].to(model.device)

        res = model.test_step(batch, 0)
        import pandas as pd
        pd.read_pickle(res, 'res.pkl')
        # print('test')
        # for batch in data_module.val_dataloader():
        #     break
        # batch['input'] = batch['input'].to(model.device)
        # batch['input_lengths'] = batch['input_lengths'].to(model.device)
        # model.test_step(batch, 0)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(__file__))
    for ckpt in ['phonemes']:
        print(ckpt)
        asr_predict(f'{base_path}/ckpt/{ckpt}')