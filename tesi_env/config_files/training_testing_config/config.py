from dataclasses import dataclass

@dataclass
class hyper_params:
    epochs: int
    lr: int
    freq_res: int
    batch_dim_train: int
    batch_dim_test: int

@dataclass
class data:
    data_name: str
    domain: str

@dataclass
class files_path:
    images: str
    csv: str

@dataclass
class models:
    net_name: str
    already_trained: bool
    path: str
    adversarial_training: bool 

@dataclass
class train_evaluation:
    par: hyper_params
    dataset: data
    paths: files_path
    model: models
