from abc import ABC, abstractmethod
from typing import Dict, List

from torch import nn


class BaseASRLoss(nn.Module, ABC):
    """ Abstract class to force ASRLosses objects to implement the following methods """

    @abstractmethod
    def forward(self, model_out: Dict, target: Dict) -> Dict:
        """
        Calculates all the loss' components. Note: The output dictionarty MUST contain a key named 'loss'. This is the
        only item that will be used in the backward propagation.
        :param model_out: the model output (the prediction).
        :param target: the gt to compare the model_out to.
        :return: a dictionary containing the loss component names and their values.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs_from_data_names(self) -> List[str]:
        """
        :return: List of all the input names needed for the loss calculations that comes from the DataLoader.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs_from_model_names(self) -> List[str]:
        """
        :return: List of the input names needed for the loss calculation that comes from the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_loss_output_keys(self) -> List[str]:
        """
        :return: list of the loss's output names (must contain an item with key 'loss'
        """
        raise NotImplementedError


# def LossesFactory(config) -> BaseASRLoss:
#     """ Return a Loss object base on the loss. type parameter in the config """
#     requested_loss = config['loss.type']
#
#     if requested_loss == 'ctc':
#         return CTCLoss(config)
#     elif requested_loss == 'asg':
#         return ASGLoss(config)
#     else:
#         raise ValueError(f" [!] Unknown loss type {requested_loss}")

#
# class CTCLoss(BaseASRLoss):
#     """CTC Loss class"""
#
#     def __init__(self, config):
#         super(CTCLoss, self).__init__()
#         self.config = config
#         self.loss_dict = {"loss": 0}
#         self.inputs_from_data_names = ['gts', 'gts_len']
#         self.inputs_from_model_names = ['preds', 'preds_len']
#         self.criterion = nn.CTCLoss()
#
#     def forward(self, model_out: Dict, target: Dict) -> Dict:
#         self.loss_dict["loss"] = self.criterion(model_out['preds'], target['gts'],
#                                                 model_out['preds_len'], target['gts_len'])
#         return self.loss_dict
#
#     def get_inputs_from_data_names(self) -> List[str]:
#         return self.inputs_from_data_names
#
#     def get_inputs_from_model_names(self) -> List[str]:
#         return self.inputs_from_model_names
#
#     def get_loss_output_keys(self) -> List[str]:
#         return list(self.loss_dict.keys())


class ASGLoss(BaseASRLoss):
    """Collection of Tacotron set-up based on provided config."""

    def __init__(self, config):
        # super(CTCLoss, self).__init__()
        self.config = config
        self.loss_dict = {"loss": 0}
        self.inputs_from_data_names = ['gts', 'gts_len']
        self.inputs_from_model_names = ['preds', 'preds_len']
        self.criterion = nn.CTCLoss()

    def forward(self, model_out: Dict, target: Dict) -> Dict:
        self.loss_dict["loss"] = self.criterion(model_out['preds'], target['gts'],
                                                model_out['preds_len'], target['gts_len'])
        return self.loss_dict

    def get_inputs_from_data_names(self) -> List[str]:
        return self.inputs_from_data_names

    def get_inputs_from_model_names(self) -> List[str]:
        return self.inputs_from_model_names

    def get_loss_output_keys(self) -> List[str]:
        return list(self.loss_dict.keys())
