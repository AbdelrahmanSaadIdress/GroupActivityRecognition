from .FullFrameNoSeq import FullFrameNoSeqDataset
from .CropsOfFullFrameButNoSeq import CropsOfFullFrameButNoSeq
from .FullFrameAndSeq import FullFrameAndSeq
from .CropsOfFullFrameAndSeq import CropsOfFullFrameAndSeq
from .HierarchicalDataSet import HierarchicalDataSet

from AnnotationsExtraction import AnnotationPreparer
from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(state: Literal["train", "val", "test"]):
    if state == "test":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    if state == "val":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    if state == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise()
            ], p=0.90),
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_dataloader(config: dict, state: Literal["train", "val", "test"]):
    annotations = AnnotationPreparer.load_annotations(config["Data"]["annotations_path"])

    # Baseline One
    if config["About"]["seq"] == "No" and config["About"]["crops"] == "No":
        return FullFrameNoSeqDataset(
            annots=annotations,
            frames_annots_path=config["Data"]["frames_annots_path"],
            transform=get_transform(state),
            matches=config["Modelling"]["data_splits"][state],
        )
    
    # Baseline Three
    ## Part1 : to classify the person activity 
    elif config["About"]["seq"] == "No" and config["About"]["crops"] == "Yes" and config["About"]["level"] == "person":
        return CropsOfFullFrameButNoSeq(
            annots=annotations,
            frames_annots_path=config["Data"]["frames_annots_path"],
            transform=get_transform(state),
            matches=config["Modelling"]["data_splits"][state],
            level="person_level"
        )
    ## Part2 : to classify the whole frame activity 
    elif config["About"]["seq"] == "No" and config["About"]["crops"] == "Yes" and config["About"]["level"] == "group":
        return CropsOfFullFrameButNoSeq(
            annots=annotations,
            frames_annots_path=config["Data"]["frames_annots_path"],
            transform=get_transform(state),
            matches=config["Modelling"]["data_splits"][state],
            level="group_level"
        )
    
    # Baseline Four
    elif config["About"]["seq"] == "Yes" and config["About"]["crops"] == "No":
        # This can be used also in the Baseline One
        return FullFrameAndSeq(
            annots=annotations,
            frames_annots_path=config["Data"]["frames_annots_path"],
            transform=get_transform(state),
            matches=config["Modelling"]["data_splits"][state],
        )
    
    # Baseline Five
    elif config["About"]["baseline"] == "5":
        if config["About"]["part"] == "a":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="person_level",
                seq=True,
                sort=False
            )
        elif config["About"]["part"] == "b":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="group_level",
                seq=True,
                sort=False
            )
    
    # Baseline Six
    elif config["About"]["baseline"] == "6":
        if config["About"]["part"] == "a":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="person_level",
                seq=False,
                sort=False
            )
        elif config["About"]["part"] == "b":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="group_level",
                seq=True,
                sort=False
            )
    
    # Baseline Seven
    elif config["About"]["baseline"] == "7"  :
        if config["About"]["part"] == "a":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="person_level",
                seq=True,
                sort=False
            )
        elif config["About"]["part"] == "b":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="group_level",
                seq=True,
                sort=False
            )

    # Baseline Eight
    elif config["About"]["baseline"] == "8"  :
        if config["About"]["part"] == "a":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="person_level",
                seq=True,
                sort=False
            )
        elif config["About"]["part"] == "b":
            return CropsOfFullFrameAndSeq(
                annots=annotations,
                frames_annots_path=config["Data"]["frames_annots_path"],
                transform=get_transform(state),
                matches=config["Modelling"]["data_splits"][state],
                level="group_level",
                seq=True,
                sort=True
            )

    # Baseline Nine
    elif config["About"]["mode"] == "Hierarical":
        return HierarchicalDataSet(
            annots=annotations,
            frames_annots_path=config["Data"]["frames_annots_path"],
            transform=get_transform(state),
            matches=config["Modelling"]["data_splits"][state],
            seq=True,
            sort=True
        )