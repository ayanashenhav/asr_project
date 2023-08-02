import torch

from .base_model import BaseModel
from typing import Dict
from torchaudio.models import Conformer
from torch import nn


class ConformerWrapper(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conformer = Conformer(**config.conformer_architecture)
        self.linear = nn.Linear(config.conformer_architecture['input_dim'], config['out_channels'])

    def inference(self, batch: Dict) -> Dict:
        return self.forward(batch)

    def forward(self, batch: Dict) -> Dict:
        preds, preds_len = self.conformer.forward(**{'input': batch['input'].permute([1, 0, 2]),
                                                 'lengths': batch['input_lengths']})
        preds = self.linear.forward(preds)
        return dict(preds=preds.permute([1, 0, 2]), preds_len=preds_len)
