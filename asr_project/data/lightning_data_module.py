import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ASRDataSet, worker_init_fn

class ASRDataModule(pl.LightningDataModule):
    def __init__(self, config, wanted_inputs):
        super().__init__()
        self.config = config
        # self.collate_fn = None
        self.train_batch_size = config['train.batch_size']
        self.wanted_inputs = wanted_inputs

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.trainset = ASRDataSet(self.config, self.config['data.train_dataset'],
                                         wanted_inputs=self.wanted_inputs)
            self.validset = ASRDataSet(self.config, self.config['data.validation_dataset'],
                                         wanted_inputs=self.wanted_inputs)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass
        if stage == "predict":
            pass

        self.print_logs()

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True
                          , num_workers=self.config['train.n_data_threads'],
                          worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.config['train.validation.batch_size'], shuffle=False,
                          num_workers=0, worker_init_fn=worker_init_fn, pin_memory=False)

    def test_dataloader(self):
        return None

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        print("\n")
        print(f"{indent}> DataLoader initialization")
        print(f"{indent}| > Train Batch size : {self.train_batch_size}")
        print(f"{indent}| > Train Dataset:")
        self.trainset.print_logs(level + 1)
        print(f"{indent}| > Validation Dataset:")
        self.validset.print_logs(level + 1)
