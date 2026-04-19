import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch import amp
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from .logger import setup_logging
from .helper import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, optimizer, criterion, scaler, dataloaders, device, config, scheduler=None, scheduler_type=None, debug=False, cont=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.dataloaders = dataloaders  # [train_loader, val_loader, test_loader]
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.debug = debug
        self.cont = cont

        # Setup experiment directory and logger
        self.exp_dir = config['Data']['root']
        os.makedirs(self.exp_dir, exist_ok=True)

        self.logger = setup_logging(self.exp_dir)
        self.logger.info(f"🚀 Starting experiment: {config['About']['name']}")
        self.logger.info(f"📦 Device: {self.device}")
        self.logger.info(f"🔢 Random seed: {config['Modelling']['seed']}")

        # TensorBoard setup
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'TENSORBOARD'))
        
        self.train_model()

    def vis(self, train_losses, val_losses, train_accs, val_accs, epoch):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(train_losses, label='Train Loss', color='blue', marker='o')
        axes[0].plot(val_losses, label='Val Loss', color='orange', marker='o')
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(train_accs, label='Train Acc', color='green', marker='o')
        axes[1].plot(val_accs, label='Val Acc', color='red', marker='o')
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True)
        axes[1].legend()
        plt.tight_layout()
        if not os.path.exists( os.path.join(self.exp_dir, "Metrics_Epoch") ):
            os.makedirs(os.path.join(self.exp_dir, "Metrics_Epoch"), exist_ok=True)
        save_path = os.path.join(self.exp_dir, "Metrics_Epoch" ,f"metrics_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"📈 Saved metrics plot to {save_path}")


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0.0, 0.0
        torch.cuda.empty_cache()

        train_loader = self.dataloaders[0]
        batch_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch:02d}")

        for batch_idx, (inputs, targets) in enumerate(batch_iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            if self.scheduler and self.scheduler_type == 'per batch':
                self.scheduler.step()
            self.scaler.update()

            total_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct += predicted.eq(targets.view(-1)).sum().item()
            total += targets.numel()

            if batch_idx % 50 == 0:
                acc = 100. * correct / total
                self.writer.add_scalar('Training/BatchLoss', loss.item(), epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('Training/BatchAccuracy', acc, epoch * len(train_loader) + batch_idx)
                self.logger.info(f"[Epoch {epoch} | Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        self.logger.info(f"✅ Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        self.writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        self.writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def val_one_epoch(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0.0, 0.0
        y_true, y_pred = [], []
        torch.cuda.empty_cache()

        val_loader = self.dataloaders[1]
        batch_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch:02d}")

        with torch.no_grad():
            for inputs, targets in batch_iterator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                total_loss += loss.item()
                predicted = outputs.argmax(-1)

                y_true.extend(targets.view(-1).cpu().numpy())
                y_pred.extend(predicted.view(-1).cpu().numpy())

                correct += predicted.eq(targets.view(-1)).sum().item()
                total += targets.numel()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        f1_val = f1_score(y_true, y_pred, average="weighted")

        self.logger.info(f"📊 Epoch {epoch} | Val Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | F1: {f1_val:.4f}")
        self.writer.add_scalar('Validation/EpochLoss', epoch_loss, epoch)
        self.writer.add_scalar('Validation/EpochAccuracy', epoch_acc, epoch)
        self.writer.add_scalar('Validation/F1Score', f1_val, epoch)

        return epoch_loss, epoch_acc, f1_val

    def train_model(self):
        best_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        if self.config['About']['preload'] == "from_start":
            start_epoch, best_acc = 0, 0.0
        else:
            start_epoch, best_acc, train_losses, val_losses, train_accs, val_accs = load_checkpoint(self.config, self.model, self.optimizer, self.scheduler, self.scaler, False, path=self.cont)
            start_epoch += 1
        
        for epoch in range(start_epoch, self.config['Modelling']['epochs']):
            self.logger.info(f"\n🚀 Epoch {epoch+1}/{self.config['Modelling']['epochs']}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, _ = self.val_one_epoch(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if self.scheduler and self.scheduler_type == 'per epoch':
                self.scheduler.step(val_loss if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"📉 Learning Rate: {current_lr:.8f}")
            self.writer.add_scalar("LR", current_lr, epoch)

            improved = val_acc > best_acc
            
            if improved:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    accuracy=val_acc,
                    loss=val_loss,
                    config=self.config,
                    is_best=improved
                )   
                best_acc = val_acc
                self.logger.info(f"🌟 New best model saved with {best_acc:.2f}% accuracy!")

            self.vis(train_losses, val_losses, train_accs, val_accs, epoch)

        self.logger.info("🎯 Training complete.")