import torch
import re

# This is not generic implementation - I don't know what we will need in the future
letter_names = {'A': 'AY',
                'B': 'BEE',
                'C': 'SEE',
                'D': 'DEE',
                'E': 'EE',
                'F': 'EFF',
                'G': 'GEE',
                'H': 'AYCH',
                'I': 'EYE',
                'J': 'JAY',
                'K': 'KAY',
                'L': 'EL',
                'M': 'EM',
                'N': 'EN',
                'O': 'OH',
                'P': 'PEE',
                'Q': 'CUE',
                'R': 'AR',
                'S': 'ESS',
                'T': 'TEE',
                'U': 'YOU',
                'V': 'VEE',
                'W': 'DOUBLE YOU',
                'X': 'EX',
                'Y': 'WHY',
                'Z': 'ZEE'}

letter_names_inv = {v: k for k, v in letter_names.items()}

letters_handling = {'pass': lambda x: x,
                    'separate_labels': lambda x: x.group().lower(),
                    'convert_to_names': lambda x: letter_names[x.group()]}


class TextTokenizer:
    def __init__(self, config):
        self.config = config
        assert self.config.letter_name_handling in letters_handling.keys(), \
            f'unknown letters_handling {self.config.letter_name_handling}'
        self.tokens = self.config.tokens
        self.labels = {t: i for i, t in enumerate(self.tokens)}
        self.blank_label = self.labels['^']

    def __call__(self, text_or_labels):
        if isinstance(text_or_labels, str):
            return self.text_to_labels(text_or_labels)
        elif isinstance(text_or_labels, list) or isinstance(text_or_labels, torch.Tensor):
            return self.labels_to_text(text_or_labels)
        else:
            raise ValueError("Expected either text or labels as input")

    def text_to_labels(self, text) -> torch.Tensor:
        text = re.sub(r'\b\w\b', letters_handling[self.config.letter_name_handling], text)
        return torch.tensor([self.labels[c] for c in text], dtype=int)

    def labels_to_text(self, labels) -> str:
        return "".join([self.tokens[c] for c in labels])

    def labels_to_text_to_eval(self, labels) -> str:
        return self.post_process_letter_name(self.labels_to_text(labels))

    def post_process_letter_name(self, text):
        if self.config.letter_name_handling == 'pass':
            return text
        if self.config.letter_name_handling == 'separate_labels':
            return text.upper()
        if self.config.letter_name_handling == 'convert_to_names':
            text = re.sub(r'\bDOUBLE YOU\b', 'W', text)
            text = re.sub(r'\b\w+\b', lambda x: letter_names_inv[x.group()], text)
            return text

    def collapse_labels(self, labels):
        collapse_labels = torch.unique_consecutive(labels)
        return collapse_labels[collapse_labels != self.blank_label]

    def from_targets_to_texts(self, targets, lens):
        i = 0
        texts = []
        for len in lens:
            texts.append(self.labels_to_text_to_eval(targets[i: i+len]))
            i += len
        return texts
