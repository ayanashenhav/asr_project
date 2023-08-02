import hydra
from typing import List, Dict

import pytorch_lightning as pl
import torch

from ..data.tokenizer import TextTokenizer
from .base_model import BaseModel
from .factory_models import ModelsFactory
# from ..loss.losses import LossesFactory
from jiwer import wer

class ASRModelLightening(BaseModel, pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModelsFactory(config.model)
        self.loss = hydra.utils.instantiate(config.loss)
        self.tokenizer = TextTokenizer(config.tokenizer)
        self.beamsearch_decoder = None
        self.init_beamsearch(config)

    def init_beamsearch(self, config):
        from torchaudio.models.decoder import ctc_decoder
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        lm_name = "convert_letters"
        self.beamsearch_decoder = ctc_decoder(lexicon=f"{base_path}/resources/kenlm/{lm_name}/lexicon.txt",
                                              tokens=f"{base_path}/resources/kenlm/{lm_name}/tokens.txt",
                                              lm=f"{base_path}/resources/kenlm/{lm_name}/kenlm.bin",)


    # def get_input_names(self) -> List[str]:
    #     return self.model.get_input_names()
    #
    # def get_output_names(self) -> List[str]:
    #     return self.model.get_output_names()
    #
    # def get_inference_input_names(self) -> List[str]:
    #     return self.model.get_inference_input_names()
    #
    # def get_inference_output_names(self) -> List[str]:
    #     return self.model.get_inference_output_names()

    def inference(self, batch: Dict) -> Dict:
        return self.model.inference(batch)

    def forward(self, batch: Dict) -> Dict:
        return self.model.forward(batch)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.model.parameters())
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.optimizer.lr)
        return optimizer

    def training_step(self, batch):
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.

        Returns:
            loss
        """
        model_out = self.model.forward(batch)
        log_probs = torch.log_softmax(model_out['preds'], dim=2)
        loss = self.loss(log_probs=log_probs, input_lengths=model_out['preds_len'],
                         targets=batch['target'], target_lengths=batch['target_lengths'])
        # model_out_for_loss = {k: v for k, v in model_out.items() if k in self.loss.get_inputs_from_model_names()}
        # data_for_loss = {k: v for k, v in batch.items() if k in self.loss.get_inputs_from_data_names()}
        # loss_dict = self.loss(model_out_for_loss, data_for_loss)
        self.log('train/loss', loss, prog_bar=True)
        # self.log_dict({'train/loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_out = self.model.inference(batch)
        log_probs = torch.log_softmax(model_out['preds'], dim=2)
        loss = self.loss(log_probs=log_probs, input_lengths=model_out['preds_len'],
                         targets=batch['target'], target_lengths=batch['target_lengths'])
        # model_out_for_loss = {k: v for k, v in model_out.items() if k in self.loss.get_inputs_from_model_names()}
        # data_for_loss = {k: v for k, v in batch.items() if k in self.loss.get_inputs_from_data_names()}
        # loss_dict = self.loss(model_out_for_loss, data_for_loss)

        pred_labels = torch.argmax(log_probs, dim=2)
        timed_preds = [self.tokenizer.labels_to_text(label) for label in pred_labels.T]
        text_preds = [self.tokenizer.labels_to_text_to_eval(self.tokenizer.collapse_labels(label)) for label in pred_labels.T]
        raw_text_preds = [self.tokenizer.labels_to_text(self.tokenizer.collapse_labels(label)) for label in
                      pred_labels.T]
        gt_texts = self.tokenizer.from_targets_to_texts(batch['target'], batch['target_lengths'])
        batch_wer = wer(gt_texts, text_preds)
        if self.current_epoch > 0 and (self.current_epoch % 50 == 0):
            beamsearch_preds = self.beamsearch_decoder(log_probs.permute([1,0,2]).detach().cpu().contiguous(),
                                                        model_out['preds_len'].detach().cpu())
            beamsearch_preds = [self.tokenizer.post_process(" ".join(p[0].words)) for p in beamsearch_preds]

            self.logger.log_text(key=f"Preds_{self.current_epoch}_epoch_{batch_wer}_wer",
                                 columns=['GT text', 'Pred text', 'Pred text (raw)', 'Beamsearch Pred Text',
                                          'Timed Argmax text'],
                                 data=[(g, t, r, b, timed) for g, t, r, b, timed in
                                       zip(gt_texts, text_preds, raw_text_preds, beamsearch_preds, timed_preds)])
            self.log('val/wer_beamsearch', wer(gt_texts, beamsearch_preds))
        # self.log_dict({'val/loss': loss, 'val/wer': batch_wer})
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/wer', batch_wer)

        return loss