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
from utils.fast_adv_training import *
from config_files.training_testing_config.config import train_evaluation
from pytorch_pretrained_vit import ViT
from torch.utils.data import random_split
from datasets import load_dataset

cs = ConfigStore.instance()
cs.store(name = 'train_eval', node = train_evaluation)

@hydra.main(config_path = 'C://Users//danie//Desktop//github tesi//tesi_env//config_files//training_testing_config', config_name = 'conv_nir_rgb', version_base='1.2')
def train_eval(cfg: train_evaluation):
    OmegaConf.to_yaml(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.data.domain == 'rgb':
        if cfg.data.data_name == 'nir':
            n_classes = 9

            # set transformations in case of training and testing
            if cfg.models.already_trained == True:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
            else:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5), 
                                      transforms.RandomCrop(224, padding = 28),
                                      transforms.ToTensor()])

            rgb = Custom_set(cfg.files_path.csv, cfg.files_path.images, augmentations)

            train_set, val_set, test_set = torch.utils.data.random_split(rgb, [330, 47, 100], generator=torch.Generator().manual_seed(1))
            # loaders
            train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg.hyper_params.batch_dim_train, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg.hyper_params.batch_dim_test, shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg.hyper_params.batch_dim_train, shuffle=False)

        if cfg.data.data_name == 'cifar':
            n_classes = 10

            # set transformations in case of training and testing
            if cfg.models.already_trained == True:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
            else:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5), 
                                      transforms.RandomCrop(224, padding = 28),
                                      transforms.ToTensor()])

            train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = augmentations)

            train_set, val_set = random_split(train_set, [45000, 5000], generator=torch.Generator().manual_seed(1))

            test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=augmentations)


            train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg.hyper_params.batch_dim_train,
                                                    shuffle=True)

            val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg.hyper_params.batch_dim_train,
                                                    shuffle=False) 

            test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg.hyper_params.batch_dim_test,
                                                    shuffle=False)
                                        
        if cfg.data.data_name == 'imagenet':
            n_classes = 200

            # set transformations in case of training and testing
            if cfg.models.already_trained == True:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
            else:
                augmentations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5), 
                                      transforms.RandomCrop(224, padding = 28),
                                      transforms.ToTensor()])

            tiny_imagenet = load_dataset('Maysee/tiny-imagenet')
            train = tiny_imagenet['train']
            val = tiny_imagenet['valid']

            train_ = Custom_set_imagenet(train, transform = augmentations)
            val_ = Custom_set_imagenet(val, transform = augmentations)

            train_set, val_set, test_set = torch.utils.data.random_split(train_, [100000, 0, 0], generator=torch.Generator().manual_seed(1))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg.hyper_params.batch_dim_train, shuffle=True)

            train_set_, val_set_, test_set_ = torch.utils.data.random_split(val_, [0, 0, 10000], generator=torch.Generator().manual_seed(1))
            test_loader = torch.utils.data.DataLoader(test_set_, batch_size = cfg.hyper_params.batch_dim_test, shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_set_, batch_size = cfg.hyper_params.batch_dim_train, shuffle=False)
            
                
    if cfg.data.domain == 'ir':
        n_classes = 9
        
        # set transformations in case of training and testing
        if cfg.models.already_trained == True:
            augmentations = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
        else:
            augmentations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5), 
                                      transforms.RandomCrop(224, padding = 28),
                                      transforms.ToTensor()])

        ir = Custom_set(cfg.files_path.NIR_ir_csv, cfg.files_path.NIR_ir, augmentations)

        train_set, val_set, test_set = torch.utils.data.random_split(ir, [330, 47, 100], generator=torch.Generator().manual_seed(1), expand=True)
        # loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg.hyper_params.batch_dim_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg.hyper_params.batch_dim_test, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg.hyper_params.batch_dim_train, shuffle=False)

    if cfg.models.net_name == 'convnext':
        model = torchvision.models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT')
        # adapting the final layer for classification
        model.classifier[2] = nn.Linear(in_features=768, out_features= n_classes)

        opt = optim.Adam(model.parameters(), lr = cfg.hyper_params.lr)
        loss = nn.CrossEntropyLoss()

        model.to(device)
        if cfg.models.already_trained == True:
            load_model(model, opt, cfg.models.path)
        
        else:
            pass

    if cfg.models.net_name == 'resnext':
        model = torchvision.models.resnext101_64x4d(weights = 'ResNeXt101_64X4D_Weights.IMAGENET1K_V1')
        # adapting the final layer for classification
        model.fc = nn.Linear(in_features=2048, out_features= n_classes, bias = True)

        opt = optim.Adam(model.parameters(), lr = cfg.hyper_params.lr)
        loss = nn.CrossEntropyLoss()

        model.to(device)
        if cfg.models.already_trained == True:
            load_model(model, opt, cfg.models.path)
        
        else:
            pass

    if cfg.models.net_name == 'vit':
        model = ViT('B_16_imagenet1k', pretrained=True, image_size = 224, num_classes = n_classes)
        # adapting the final layer for classification
        model.fc = nn.Linear(in_features=768, out_features=n_classes, bias=True)

        opt = optim.Adam(model.parameters(), lr = cfg.hyper_params.lr)
        loss = nn.CrossEntropyLoss()

        model.to(device)
        if cfg.models.already_trained == True:
            load_model(model, opt, cfg.models.path)
        
        else:
            pass

    if cfg.models.already_trained == True:
        # testing the model if already trained
        test(model, test_loader)
    
    if cfg.models.already_trained == False:
        # else train the model
        for epoch in range(cfg.hyper_params.epochs):
            train(epoch, train_loader, model, opt, loss, cfg.hyper_params.freq_res)
            validation(model, val_loader, loss)
    
    if cfg.models.adversarial_training == True:
        # adversarially training a model
        for epoch in range(cfg.hyper_params.epochs):
            adv_train(epoch, train_loader, model, opt, loss, cfg.hyper_params.freq_res)

if __name__ == '__main__':
    train_eval()