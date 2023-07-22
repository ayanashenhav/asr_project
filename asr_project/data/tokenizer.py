import torch

# This is not generic implementation - I don't know what we will need in the future

class TextTokenizer:
    def __init__(self, config):
        self.config = config
        assert self.config.method == 'char', 'only char tokenization is supported'
        self.tokens = self.config.tokens
        self.labels = {t: i for i, t in enumerate(self.tokens)}

    def __call__(self, text_or_labels):
        if isinstance(text_or_labels, str):
            return self.text_to_labels(text_or_labels)
        elif isinstance(text_or_labels, list) or isinstance(text_or_labels, torch.Tensor):
            return self.labels_to_text(text_or_labels)
        else:
            raise ValueError("Expected either text or labels as input")

    def text_to_labels(self, text) -> torch.Tensor:
        return torch.tensor([self.labels[c] for c in text], dtype=int)

    def labels_to_text(self, labels) -> str:
        return "".join([self.tokens[c] for c in labels])
