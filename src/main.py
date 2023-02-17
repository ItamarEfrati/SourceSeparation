from src.musdb_data_module import MUSDBDataModule

if __name__ == '__main__':
    d = MUSDBDataModule()
    d.setup()
    t_d = d.train_dataloader()
    print(1)