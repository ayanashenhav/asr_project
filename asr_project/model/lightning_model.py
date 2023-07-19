import os
from collections import Counter
from typing import List, Dict

import pytorch_lightning as pl

from .base_model import BaseModel
from .factory_models import ModelsFactory
from ..loss.losses import LossesFactory
from ..optimizer.optimizer import get_optimizer


class ASRModelLightening(BaseModel, pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModelsFactory(config)
        self.loss = LossesFactory(config)

    def get_input_names(self) -> List[str]:
        return self.model.get_input_names()

    def get_output_names(self) -> List[str]:
        return self.model.get_output_names()

    def get_inference_input_names(self) -> List[str]:
        return self.model.get_inference_input_names()

    def get_inference_output_names(self) -> List[str]:
        return self.model.get_inference_output_names()

    def inference(self, batch: Dict) -> Dict:
        return self.model.inference(batch)

    def forward(self, batch: Dict) -> Dict:
        return self.model.forward(batch)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.config['train.optimizer'])
        return optimizer

    def training_step(self, batch):
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.

        Returns:
            Tuple[Dict, Dict]: Model outputs and computed losses.
        """

        return_dict = {}
        input_dict = {k: v for k, v in batch.items() if k in self.get_input_names()}
        model_out = self.model.forward(input_dict)
        model_out_for_loss = {k: v for k, v in model_out.items() if k in self.loss.get_inputs_from_model_names()}
        data_for_loss = {k: v for k, v in batch.items() if k in self.loss.get_inputs_from_data_names()}
        loss_dict = self.loss(model_out_for_loss, data_for_loss)
        return_dict['loss'] = loss_dict['loss']
        return return_dict

    def validation_step(self, batch, batch_idx):

        input_dict = {k: v for k, v in batch.items() if k in self.get_input_names()}
        model_out = self.model.inference(input_dict)
        model_out_for_loss = {k: v for k, v in model_out.items() if k in self.loss.get_inputs_from_model_names()}
        data_for_loss = {k: v for k, v in batch.items() if k in self.loss.get_inputs_from_data_names()}
        loss_dict = self.loss(model_out_for_loss, data_for_loss)
        val_results = {'model_out': model_out, 'losses': loss_dict}
        return {'batches': batch, 'val_results': val_results}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loss_name = self.config['loss.type']
        self.log(f'train/{loss_name}_loss', outputs['loss'])

    def validation_epoch_end(self, outputs):
        # Validation logging
        if not len(outputs):
            return

        tot_loss = Counter()
        for res in outputs:
            tot_loss += Counter(res['val_results']['losses'])
        mean_loss = {k: v / len(outputs) for k, v in tot_loss.items()}
        if 'loss' not in mean_loss.keys():
            mean_loss['loss'] = 0.0

        self._log('val/', data=mean_loss)

    def _log(self, base_dir: str, data: Dict):
        """Saves the data to the logger"""
        for k, v in data.items():
            self.log(os.path.join(base_dir, k), v, prog_bar=(k == 'loss'), sync_dist=True)
