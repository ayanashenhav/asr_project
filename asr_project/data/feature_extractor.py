import torchaudio
import torch
import librosa

class FeatureExtractor:
    def __init__(self, sample_rate, n_mfcc, melkwargs, compute_deltas):

        self.compute_deltas = compute_deltas
        self.sample_rate=sample_rate

        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)
        if compute_deltas:
            self.compute_deltas = torchaudio.transforms.ComputeDeltas()

    def __call__(self, raw_wav):
        mfcc_res = self.mfcc(raw_wav)
        if self.compute_deltas:
            delta_one = self.compute_deltas(mfcc_res)
            delta_two = self.compute_deltas(delta_one)
            mfcc_res = torch.vstack([mfcc_res, delta_one, delta_two])

        return mfcc_res
