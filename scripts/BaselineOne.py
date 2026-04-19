import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import  GradScaler
from torch.utils.data import DataLoader

from utils import set_seed, load_config

from DataLoaders import get_dataloader
from networks import FramesModel


def prepare_the_run(config):
    
    set_seed(config['Modelling']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = get_dataloader(config, "train" )
    train_dataloader = DataLoader(train_dataset, batch_size = config['Modelling']['batch_size'], shuffle = True, num_workers=4, pin_memory=True)
    
    val_dataset = get_dataloader(config, "val" )
    val_dataloader = DataLoader(val_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)
    
    test_dataset = get_dataloader(config, "test" )
    test_dataloader = DataLoader(test_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)

    dataloaders = [train_dataloader, val_dataloader, test_dataloader]

    model = FramesModel(config['Modelling']['num_classes']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['Modelling']['lr'], weight_decay=config['Modelling']['weight_decay'] )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=5, verbose=True)
    scheduler_type = 'per epoch'

    # counts = torch.tensor([train_dataset.labels_count[cls] for cls in train_dataset.labels.keys()])
    # N = counts.sum()
    # K = len(counts)
    # weights = N / (K * counts)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(device))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    epochs = config['Modelling']['epochs']

    return model, device, dataloaders, optimizer, scheduler, scheduler_type, criterion, scaler, epochs, config, train_dataset
