from dataclasses import dataclass

@dataclass
class hyper_params:
    epochs: int
    gen_lr: float
    dis_lr: float
    batch_dim: int

@dataclass
class files_path:
    clean_images: str
    clean_csv: str
    adv_images: str
    adv_csv: str

@dataclass
class train_gen:
    par: hyper_params
    paths: files_path