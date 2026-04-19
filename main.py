import argparse
from utils import load_config
from scripts import prepare_the_run_B9
from utils import Trainer, Tester
from AnnotationsExtraction.Annotations import BoxInfo

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Baseline_One.yaml", help="Path to the config file")
    parser.add_argument("--t", type=str, default="train", help="train or test")
    parser.add_argument("--pt", type=str, default=None, help="path of the model to use it in testing")

    args = parser.parse_args()

    config = load_config(args.config)
    t = args.t
    pt = args.pt

    model, device, dataloaders, optimizer, scheduler, scheduler_type, criterion, scaler, epochs, config, train_dataset = prepare_the_run_B9(config)

    if t == "train":
        Trainer(model, optimizer, criterion, scaler, dataloaders, device, config, scheduler, scheduler_type)
    elif t == "test":
            Tester(config, model,optimizer, scheduler, scaler,  criterion,
                dataloaders, device, 
                class_names=list(train_dataset.group_activity_labels_count.keys()),
                path = pt
            )