import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import  GradScaler
from torch.utils.data import DataLoader

from utils import set_seed, load_config, load_checkpoint

from DataLoaders import get_dataloader
from networks import OneCropNoSeqModel, WholeCropsNoSeqModel

import os

def prepare_the_run_B3_P1(config):
    
    set_seed(config['Modelling']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = get_dataloader(config, "train" )
    train_dataloader = DataLoader(train_dataset, batch_size = config['Modelling']['batch_size'], shuffle = True, num_workers=4, pin_memory=True)
    
    val_dataset = get_dataloader(config, "val" )
    val_dataloader = DataLoader(val_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)
    
    test_dataset = get_dataloader(config, "test" )
    test_dataloader = DataLoader(test_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)

    dataloaders = [train_dataloader, val_dataloader, test_dataloader]

    model = OneCropNoSeqModel(config['Modelling']['num_classes']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['Modelling']['lr'], weight_decay=config['Modelling']['weight_decay'] )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=5, verbose=True)
    scheduler_type = 'per epoch'


    # # determine which label/count maps the dataset provides
    # if hasattr(train_dataset, "person_activity_labels") and hasattr(train_dataset, "person_activity_labels_count"):
    #     labels_map = train_dataset.person_activity_labels          # e.g. {"waiting":0, "setting":1, ...}
    #     counts_map = train_dataset.person_activity_labels_count    # e.g. {"waiting": 123, ...}
    # elif hasattr(train_dataset, "group_activity_labels") and hasattr(train_dataset, "group_activity_labels_count"):
    #     labels_map = train_dataset.group_activity_labels
    #     counts_map = train_dataset.group_activity_labels_count
    # else:
    #     labels_map = None
    #     counts_map = None

    # if labels_map is not None and counts_map is not None:
    #     # build counts tensor in the order of label indices
    #     # sort by index so the counts align with class indices used by the model
    #     sorted_items = sorted(labels_map.items(), key=lambda kv: kv[1])  # [(class_name, idx), ...] sorted by idx
    #     counts_list = [counts_map[class_name] for class_name, _ in sorted_items]
    #     counts = torch.tensor(counts_list, dtype=torch.float32, device=device)

    #     # avoid division by zero for any rare class
    #     eps = 1e-6
    #     counts = counts.clamp(min=eps)

    #     N = counts.sum()
    #     K = counts.numel()
    #     weights = N / (K * counts)   # inverse-frequency weighting

    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    # else:
    #     # fallback: no class-count info available
    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # # --- end balanced weights block ---

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    epochs = config['Modelling']['epochs']

    return model, device, dataloaders, optimizer, scheduler, scheduler_type, criterion, scaler, epochs, config, train_dataset




def prepare_the_run_B3_P2(config ,ck_path):
    
    set_seed(config['Modelling']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = get_dataloader(config, "train" )
    train_dataloader = DataLoader(train_dataset, batch_size = config['Modelling']['batch_size'], shuffle = True, pin_memory=True, drop_last=True, persistent_workers=True, num_workers = min(8, os.cpu_count() // 2), pin_memory_device='cuda')  
    
    val_dataset = get_dataloader(config, "val" )
    val_dataloader = DataLoader(val_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)
    
    test_dataset = get_dataloader(config, "test" )
    test_dataloader = DataLoader(test_dataset, batch_size = config['Modelling']['batch_size'], shuffle = False, num_workers=4, pin_memory=True)

    dataloaders = [train_dataloader, val_dataloader, test_dataloader]

    model1 = OneCropNoSeqModel().to(device)
    if ck_path:
        model_path = ck_path
        load_checkpoint(config, model1, None, False, model_path )

    model = WholeCropsNoSeqModel(model1, config['Modelling']['num_classes']).to(device)

    # optimizer = optim.AdamW([{"params": model.feature_extraction[-1].parameters(), "lr": 1e-5},
    #                         {"params": model.fc.parameters(), "lr": config['Modelling']['lr']}],
    #                         lr=config['Modelling']['lr'], weight_decay=config['Modelling']['weight_decay'] )
    optimizer = optim.AdamW(model.parameters(), lr=config['Modelling']['lr'], weight_decay=config['Modelling']['weight_decay'] )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=5, verbose=True)
    scheduler_type = 'per epoch'

    # determine which label/count maps the dataset provides
    # if hasattr(train_dataset, "person_activity_labels") and hasattr(train_dataset, "person_activity_labels_count"):
    #     labels_map = train_dataset.person_activity_labels          # e.g. {"waiting":0, "setting":1, ...}
    #     counts_map = train_dataset.person_activity_labels_count    # e.g. {"waiting": 123, ...}
    # if hasattr(train_dataset, "group_activity_labels") and hasattr(train_dataset, "group_activity_labels_count"):
    #     labels_map = train_dataset.group_activity_labels
    #     counts_map = train_dataset.group_activity_labels_count
    # else:
    #     labels_map = None
    #     counts_map = None

    # if labels_map is not None and counts_map is not None:
    #     # build counts tensor in the order of label indices
    #     # sort by index so the counts align with class indices used by the model
    #     sorted_items = sorted(labels_map.items(), key=lambda kv: kv[1])  # [(class_name, idx), ...] sorted by idx
    #     counts_list = [counts_map[class_name] for class_name, _ in sorted_items]
    #     counts = torch.tensor(counts_list, dtype=torch.float32, device=device)

    #     # avoid division by zero for any rare class
    #     eps = 1e-6
    #     counts = counts.clamp(min=eps)

    #     N = counts.sum()
    #     K = counts.numel()
    #     weights = N / (K * counts)   # inverse-frequency weighting

    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    # else:
    #     # fallback: no class-count info available
    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # --- end balanced weights block ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    epochs = config['Modelling']['epochs']

    return model, device, dataloaders, optimizer, scheduler, scheduler_type, criterion, scaler, epochs, config, train_dataset
