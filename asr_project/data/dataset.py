from functools import wraps
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.functional import resample

# dataset functions

class ASRDataSet(Dataset):
    def __init__(self, config, dataset_args, wanted_inputs):
        super().__init__()
        self.config = config
        self.dataset_args = dataset_args
        self.wanted_inputs = wanted_inputs
        max_length_sec = self.dataset_args['max_length_data_sec']
        target_sample_rate = config['train.target_sample_rate']
        max_length = max_length_sec * target_sample_rate
        seq_len_multiple_of = config.get('model.modules.codec.seq_len_multiple_of', None) # Todo: Fix this.
        self.filelist = self.dataset_args['filelist']
        self.validation = self.dataset_args['validation']

        files = []
        if not isinstance(self.filelist, list):
            self.filelist = [self.filelist]

        for file in self.filelist:
            lst_wav = list(open(file, 'r').read().split('\n'))
            lst_wav.remove("")
            files.extend(lst_wav)

        assert len(files) > 0, 'no sound files found'

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        num_outputs = len(self.target_sample_rate)
        data = cast_tuple(data, num_outputs)

        # resample if target_sample_rate is not None in the tuple

        data_tuple = tuple(
            (resample(d, sample_hz, target_sample_rate) if exists(target_sample_rate) else d) for d, target_sample_rate
            in zip(data, self.target_sample_rate))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            # pad or curtail

            if audio_length > max_length:
                max_start = int(audio_length - max_length)
                start = torch.randint(0, max_start, (1,)) if not self.validation else 0
                data = data[:, start:start + max_length]

            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

            data = rearrange(data, '1 ... -> ...')

            if exists(max_length):
                data = data[:max_length]

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]
        return output

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print(f"{indent}> Dataset initialization")
        print(f"{indent}| > Is validation dataset : {self.validation}")
        print(f"{indent}| > Numbe of samples : {self.__len__()}")
        print(f"{indent}| > Wavs filelist : {self.filelist}")
        print(f"{indent}| > Target Sample Rate : {self.target_sample_rate}")
        print(f"{indent}| > Max length data : {self.max_length}")


# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):

        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)

            return {'orig_audio': data[:, None, :]}

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return {'orig_audio': tuple(outputs)}

    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):

    return {'orig_audio': pad_sequence(data, batch_first=True)}


def worker_init_fn(worker_id):
    """
    Utility function to be used in dataloaders to allow working with indexed binaries in workers
    For details, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    np.random.seed(worker_id * 100)
