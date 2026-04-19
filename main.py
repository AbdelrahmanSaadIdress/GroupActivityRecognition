import argparse
from utils import load_config
from scripts import prepare_the_run_B8
from utils import Trainer, Tester
from AnnotationsExtraction.Annotations import BoxInfo

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Baseline_One.yaml", help="Path to the config file")
    parser.add_argument("--p", type=str, default="part1", help="Which part of the baseline")
    parser.add_argument("--t", type=str, default="train", help="train or test")
    
    parser.add_argument("--cp", type=str, default=None, help="path of the Baseline 3 model")
    parser.add_argument("--cpp", type=str, default=None, help="path of the first part of the baseline for the second part which is Part A of Baseline five")
    parser.add_argument("--pt", type=str, default=None, help="path of the model to use it in testing")
    parser.add_argument("--cont", type=str, default=None, help="path of the model to continue")

    args = parser.parse_args()

    config = load_config(args.config)

    p = args.p
    t = args.t
    cp = args.cp
    cpp = args.cpp
    pt = args.pt
    cont = args.cont


    model, device, dataloaders, optimizer, scheduler, scheduler_type, criterion, scaler, epochs, config, train_dataset = prepare_the_run_B8(config, cpp)


    if t == "train":
        Trainer(model, optimizer, criterion, scaler, dataloaders, device, config, scheduler, scheduler_type, False, cont=cont)

    elif t == "test":
            Tester(config, model,optimizer, scheduler, scaler,  criterion,
                dataloaders, device, 
                class_names=list(train_dataset.group_activity_labels_count.keys()),
                path = pt
            )