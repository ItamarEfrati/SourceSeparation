from musdb_data_module import MUSDBDataModule

if __name__ == '__main__':
    d = MUSDBDataModule()
    d.prepare_data()
    d.setup()
    next((d.train_dataloader().__iter__()))
    next((d.val_dataloader().__iter__()))
    next((d.test_dataloader().__iter__()))
    print(1)
