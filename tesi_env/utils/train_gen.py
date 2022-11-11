import sys
sys.path.append('C://Users//danie//Desktop//github tesi//tesi_env//config_files')
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, models, transforms
from utils.custom_set import *
from utils.eval import *
from utils.save_load_model import *
from utils.generator import *
from config_files.training_testing_config.config import train_gen
from pytorch_pretrained_vit import ViT
from torch.utils.data import random_split
from datasets import load_dataset

cs = ConfigStore.instance()
cs.store(name = 'train_eval', node = train_evaluation)

@hydra.main(config_path = 'C://Users//danie//Desktop//github tesi//tesi_env//config_files//train_gen_config', config_name = 'conv_nir_rgb', version_base='1.2')
def train_generator(cfg: train_gen):
    OmegaConf.to_yaml(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 9

    augmentations = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])

    double_set = Double_custom_set(cfg.files_path.clean_csv, cfg.files_path.adv_csv,
             cfg.files_path.clean_images, cfg.files_path.adv_images, augmentations)

    train_set, val_set, test_set = torch.utils.data.random_split(double_set, [477, 0, 0], generator=torch.Generator().manual_seed(1))
    # loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg.hyper_params.batch_dim, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg.hyper_params.batch_dim, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg.hyper_params.batch_dim, shuffle=False)

    gen = Large_generator(3).to(device)
    dis = Large_Discriminator(3).to(device)
    opt_gen = optim.Adam(gen.parameters(), lr = cfg.hyper_params.gen_lr)
    opt_dis = optim.Adam(dis.parameters(), lr = cfg.hyper_params.dis_lr)
    binary_ce = nn.BCELoss()
    mse_loss = nn.MSELoss()
   
    train_generative_model(gen, dis, binary_ce, mse_loss, train_loader)

if __name__ == '__main__':
    train_generator()