from typing import Dict, List

import torch
from torch import nn

from .base_model import BaseModel
from ..layers.res_conv_bn import Conv1dBNBlock

class CNNModel(nn.Module, BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config['model.in_channels']
        hidden_channels = config['model.hidden_channels']
        out_channels = config['model.out_channels']
        kernel_size = config['model.kernel_size']
        num_conv_blocks = config['model.num_conv_blocks']

        self.net = nn.Sequential(
            Conv1dBNBlock(in_channels, hidden_channels, hidden_channels, kernel_size, 1,
                          num_conv_blocks=num_conv_blocks),
            nn.Conv1d(hidden_channels, out_channels, 1), )

    def get_input_names(self) -> List[str]:
        if self.config['model.mel']:
            return ['mel']
        elif self.config['model.mfcc']:
            return ['mfcc']

    def get_output_names(self) -> List[str]:
        return ['preds', 'preds_len']

    def get_inference_input_names(self) -> List[str]:
        return self.get_input_names()

    def get_inference_output_names(self) -> List[str]:
        return self.get_output_names()

    def inference(self, batch: Dict) -> Dict:
        return self.forward(batch)

    def forward(self, batch: Dict) -> Dict:
        preds = self.net(batch[self.feature])
        lens = torch.ones(preds.shape[0]) * preds.shape[1]
        return {'preds': preds, 'preds_len': lens}
