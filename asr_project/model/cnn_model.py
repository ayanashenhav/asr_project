from typing import Dict, List
from torch import nn

from .base_model import BaseModel
from ..layers.res_conv_bn import Conv1dBNBlock, ResidualConv1dBNBlock


class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config.architecture['in_channels']
        hidden_channels = config.architecture.hidden_channels
        out_channels = config.architecture.out_channels
        kernel_size = config.architecture.kernel_size
        num_conv_blocks = config.architecture.num_conv_blocks
        num_res_blocks = config.architecture.num_res_blocks
        p = config.architecture.p
        if num_res_blocks is None:
            self.net = nn.Sequential(
                Conv1dBNBlock(in_channels, hidden_channels, hidden_channels, kernel_size, 1, num_conv_blocks=num_conv_blocks, p=p),
                nn.Conv1d(hidden_channels, out_channels, 1))
        else:
            self.net = nn.Sequential(
                Conv1dBNBlock(in_channels, hidden_channels, hidden_channels, kernel_size, 1, num_conv_blocks=1),
                ResidualConv1dBNBlock(hidden_channels, hidden_channels, hidden_channels, kernel_size, num_res_blocks*[1], num_conv_blocks=num_conv_blocks, num_res_blocks=num_res_blocks, p=p),
                nn.Conv1d(hidden_channels, out_channels, 1))
        t=1


    def get_input_names(self) -> List[str]:
        return ['input', 'input_lengths']

    def get_output_names(self) -> List[str]:
        return ['preds', 'preds_len']

    def get_inference_input_names(self) -> List[str]:
        return self.get_input_names()

    def get_inference_output_names(self) -> List[str]:
        return self.get_output_names()

    def inference(self, batch: Dict) -> Dict:
        return self.forward(batch)

    def forward(self, batch: Dict) -> Dict:
        return {'preds': self.net(batch['input'].permute([1, 2, 0])).permute([2, 0, 1]),
                'preds_len': batch['input_lengths']}

