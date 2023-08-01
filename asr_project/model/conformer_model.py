import torch

from .base_model import BaseModel
from typing import Dict
from torchaudio.models import Conformer


class ConformerWrapper(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Conformer(**config.architecture)

    def inference(self, batch: Dict) -> Dict:
        self.model.eval()
        with torch.no_grad():
            return self.forward(batch)

    def forward(self, batch: Dict) -> Dict:
        preds, preds_len = self.model.forward(**{'input': batch['input'].permute([1, 0, 2]),
                                                 'lengths': batch['input_lengths']})
        return dict(preds=preds.permute([1, 0, 2]), preds_len=preds_len)
