from dataclasses import dataclass

@dataclass
class hyper_params:
    epochs: int
    lr: int
    freq_res: int
    batch_dim_train: int
    batch_dim_test: int

@dataclass
class hyper_params_attack:
    attack: str
    x_dim: int
    y_dim: int
    restarts: int
    iter_per_restart: int
    perc_pixel_per_iter : int
    number_att : int
    return_stats : bool

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

@dataclass
class test_att:
    par: hyper_params
    att_par : hyper_params_attack
    dataset: data
    paths: files_path
    model: models
