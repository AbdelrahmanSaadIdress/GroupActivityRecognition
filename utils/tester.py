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
# from torch.cuda.amp import autocast, GradScaler
from torch import amp
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from .logger import setup_logging
from .helper import save_checkpoint, load_checkpoint


class Tester:
    def __init__(self, config, model, optimizer,scheduler,scaler, 
                criterion, dataloaders, device, class_names=[], save_dir='/kaggle/working/Group-Activity-Recognition/src/Outputs/CONFUSION_MATRIX', path = None):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        

        _ = load_checkpoint(self.config, self.model, None, None, None, test = True, path = path)
        t = self.test(dataloaders[2], prefix = config['About']['description'])

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            print(f'🧩 Confusion matrix saved at: {save_path}')
        plt.close(fig)

    def test(self, dataloader, prefix=''):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss, correct, total = 0.0, 0.0, 0.0
        torch.cuda.empty_cache()

        batch_iterator = tqdm(dataloader, desc=f"{prefix} Testing")
        with torch.no_grad():
            for inputs, targets in batch_iterator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)) if self.criterion else 0

                total_loss += loss.item() if self.criterion else 0
                predicted = outputs.argmax(-1)

                y_true.extend(targets.view(-1).cpu().numpy())
                y_pred.extend(predicted.view(-1).cpu().numpy())

                correct += predicted.eq(targets.view(-1)).sum().item()
                total += targets.numel()

        acc = 100. * correct / total
        avg_loss = total_loss / len(dataloader) if self.criterion else 0
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("\n" + "=" * 60)
        print(f"{prefix} Evaluation Summary")
        print("=" * 60)
        print(f"Accuracy     : {acc:.2f}%")
        print(f"Loss         : {avg_loss:.4f}")
        print(f"F1 (weighted): {f1:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        if self.class_names:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
            self.plot_confusion_matrix(y_true, y_pred, self.class_names, save_path)

        return {"accuracy": acc, "loss": avg_loss, "f1": f1}
