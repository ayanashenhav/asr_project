import torch.nn as nn
from typing import List
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """ Abstract class to force Models to implement the following methods """
    @abstractmethod
    def get_input_names(self) -> List[str]:
        """
        :return: A list of the model's input components names.
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_names(self) -> List[str]:
        """
        :return: A list of the model's output components names.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inference_input_names(self) -> List[str]:
        """
        :return: A list of the model's input component names needed for prediction (inference).
        """
        raise NotImplementedError

    @abstractmethod
    def get_inference_output_names(self) -> List[str]:
        """
        :return: A list of the model's component names that are outputted from inference.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abstractmethod
    def inference(self, inputs):
        raise NotImplementedError

