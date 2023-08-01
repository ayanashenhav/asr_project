import os.path
import numpy as np
import hydra
import torch
import torchaudio
from torch.utils.data import Dataset
from .tokenizer import TextTokenizer


base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dataset_files = dict(train=os.path.join(base_path, 'resources/data_files/train.csv'),
                     validation=os.path.join(base_path, 'resources/data_files/validation.csv'),
                     test=os.path.join(base_path, 'resources/data_files/test.csv'))


class ASRDataSet(Dataset):
    def __init__(self, config, mode, ):
        super().__init__()
        self.config = config.data
        self.mode = mode
        assert mode in dataset_files.keys(), f'unknown mode {mode}'
        self.filelist = dataset_files[mode]

        # Preparing samples in init.
        samples = self.read_samples()
        self.samples = samples

        self.tokenizer = TextTokenizer(config.tokenizer)
        self.feature_extractor = hydra.utils.instantiate(config.feature_extractor.cls)
        self.output_per_sec = self.feature_extractor(torch.rand(self.feature_extractor.sample_rate)).shape[1]
        if self.config['prepare_data_on_init']:
            for sample in self.samples:
                sample.update({'input': self.feature_extractor(sample['raw_wav']).T,
                                           'target': self.tokenizer(sample['raw_text'])})

    def read_samples(self):
        samples = []
        lines = open(self.filelist, 'r').readlines()
        for line in lines:
            path, text = line.split(',')
            audio, fs = torchaudio.load(os.path.join(self.config.audio_base_path, path))
            assert fs == self.config.fs, f'expected sample rate of {self.config.fs} but got {fs}'
            assert audio.shape[0] == 1
            samples.append({'sample_id': path.split('/')[-1].split(".")[0],
                            'raw_wav': audio[0],
                            'raw_text': text.strip()})
        assert len(samples) > 0, 'no sound samples found'
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.config['prepare_data_on_init']:
            sample['input'] = self.feature_extractor(sample['raw_wav']).T
            sample['target'] = self.tokenizer(sample['raw_text'])
        if self.config['augmentations']['add_random_silence'] and self.mode == 'train':

            if np.random.rand() < 0.2:
                sample['input'] = torch.cat([sample['input'],
                                             torch.zeros(np.random.randint(0, self.output_per_sec//2),
                                                         sample['input'].shape[1])])
        return sample

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print(f"{indent}> Dataset initialization")
        print(f"{indent}| > Dataset mode : {self.mode}")
        print(f"{indent}| > Number of samples : {self.__len__()}")
        print(f"{indent}| > Sampels filelist : {self.filelist}")
        # print(f"{indent}| > Sample Rate : {self.target_sample_rate}")


# dataloader functions
def worker_init_fn(worker_id):
    """
    Utility function to be used in dataloaders to allow working with indexed binaries in workers
    For details, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    np.random.seed(worker_id * 100)


def collate_fn(samples):
    # the collate function assumes that samples contain "input" and "target" keys only
    # stack all inputs, pad to the longest sequence and product input_legnths contains the original lengths
    inputs = [s['input'] for s in samples]
    input_lengths = torch.LongTensor([len(s) for s in inputs])
    inputs = torch.nn.utils.rnn.pad_sequence(inputs,)
    # concatenate all targets, and create target_legnths contains the original lengths
    targets = [s['target'] for s in samples]
    target_lengths = torch.LongTensor([len(s) for s in targets])
    targets = torch.cat(targets)

    batch = {'input': inputs, 'target': targets, 'input_lengths': input_lengths, 'target_lengths': target_lengths}
    return batch