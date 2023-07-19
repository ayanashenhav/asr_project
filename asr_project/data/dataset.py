import os.path
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class ASRDataSet(Dataset):
    def __init__(self, config, dataset_args, wanted_inputs):
        super().__init__()
        self.config = config
        self.dataset_args = dataset_args
        self.wanted_inputs = wanted_inputs
        self.filelist = self.dataset_args['filelist']
        self.validation = self.dataset_args['validation']

        # Preparing samples in init.
        # todo: consider reading the audio and normalize the text here, so i won't happen more than once.
        samples = []
        lines = list(open(self.filelist, 'r').read().split('\n'))
        lines.remove("")
        for line in lines:
            path, text = line.split('|')
            samples.append({'wav_path': os.path.abspath(path), 'text': text})

        assert len(samples) > 0, 'no sound samples found'

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item_data = dict()
        raw_wav, sample_hz = torchaudio.load(sample['wav_path'])

        if raw_wav.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            raw_wav = torch.mean(raw_wav, dim=0).unsqueeze(0)

        item_data['raw_wav'] = raw_wav
        item_data['raw_text'] = sample['text']

        # TODO: Add here some feature extration.
        # TODO: Need to toknize the tex to labels (ids?) so it can be pad for the loss
        # Extract Audio Features
        if 'mel' in self.wanted_inputs:
            pass
        if 'mfcc' in self.wanted_inputs:
            pass

        # Extract Text Features
        if 'phonemes' in self.wanted_inputs:
            pass

        return item_data

    def collate_fn(self, batch):
        pass
        # TODO: Add padding to each relevant item.

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print(f"{indent}> Dataset initialization")
        print(f"{indent}| > Is validation dataset : {self.validation}")
        print(f"{indent}| > Numbe of samples : {self.__len__()}")
        print(f"{indent}| > Sampels filelist : {self.filelist}")
        # print(f"{indent}| > Sample Rate : {self.target_sample_rate}")


# dataloader functions
def worker_init_fn(worker_id):
    """
    Utility function to be used in dataloaders to allow working with indexed binaries in workers
    For details, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    np.random.seed(worker_id * 100)
