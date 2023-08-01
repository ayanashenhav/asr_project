import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ASRDataSet, worker_init_fn, collate_fn


class ASRDataModule(pl.LightningDataModule):
    def __init__(self, config, ):
        super().__init__()
        self.config = config
        self.train_batch_size = config.data.dataloader.train_batch_size
        self.validation_batch_size = config.data.dataloader.validation_batch_size

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.trainset = ASRDataSet(self.config, mode='train')
            self.validset = ASRDataSet(self.config, mode='validation')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass
        if stage == "predict":
            self.trainset = ASRDataSet(self.config, mode='train')
            self.validset = ASRDataSet(self.config, mode='validation')

        self.print_logs()

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.config.data.dataloader.n_workers, collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.validation_batch_size, shuffle=False,
                          num_workers=0, worker_init_fn=worker_init_fn, pin_memory=False,
                          collate_fn=collate_fn)

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
