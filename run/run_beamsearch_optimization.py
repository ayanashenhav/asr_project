import os
from omegaconf import OmegaConf
import pandas as pd
from jiwer import wer
from torchaudio.models.decoder import ctc_decoder
from asr_project.data.tokenizer import TextTokenizer

config = OmegaConf.load('/home/vpnuser/cs_huji/speech/asr_project/ckpt/phonemes/config.yaml')
tokenizer = TextTokenizer(config.tokenizer)
gt_texts, bs_params = pd.read_pickle("/home/vpnuser/Downloads/res.pkl")


def get_wer(kwargs):
    bs_decoder = ctc_decoder(lexicon="/home/vpnuser/cs_huji/speech/asr_project/resources/kenlm/phonemes/lexicon.txt",
                             tokens="/home/vpnuser/cs_huji/speech/asr_project/resources/kenlm/phonemes/tokens.txt",
                             lm="/home/vpnuser/cs_huji/speech/asr_project/resources/kenlm/phonemes/kenlm_2.bin",
                             **kwargs)
    bs_texts = bs_decoder(*bs_params)
    bs_texts = [tokenizer.post_process(" ".join(t[0].words)) for t in bs_texts]
    wer_ = wer(gt_texts, bs_texts)
    print(round(100*wer_, 2))
    return 100*wer_


if __name__ == '__main__':
    get_wer({})
    print({word_score: get_wer({'word_score': word_score}) for word_score in [0.1, 0.2, 0.5, 1, 2]})
    print({lm_weight: get_wer({'lm_weight': lm_weight}) for lm_weight in [0.2, 0.5, 1, 2, 3, 4]})