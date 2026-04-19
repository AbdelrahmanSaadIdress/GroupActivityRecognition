import os 
import cv2
import torch
from torch.utils.data import Dataset

from typing import List

class HierarchicalDataSet(Dataset):
    def __init__(self, annots, frames_annots_path, transform, matches:List[int], seq:bool=True,sort:bool = True):
        """
        Args:
            annots (dict): Dictionary containing annotations organized by match and clip.
            frames_annots_path (str): Root path where frame images are stored.
            transform (albumentations.Compose): Albumentations transform pipeline to apply on frames.
            matches (list): List of match IDs (as strings or integers) to include in this dataset.
            level (string): Determine whether one crop of player or the whole crops will be shown .
                if "person_level" then one crop with the person activity
                if "group_level" then 12 crops with the group activity

        """
        self.annots = annots 
        self.video_path = frames_annots_path
        self.transform = transform
        self.sort = sort
        self.seq = seq 
        
        # Define the fixed set of group activity classes and map them to numeric labels
        person_activity_classes = [
            "Waiting", "Setting", "Digging",
            "Falling", "Spiking", "Blocking",
            "Jumping", "Moving", "Standing"
        ]
        person_activity_classes = [ cls.strip().lower() for cls in person_activity_classes]
        group_activity_classes = [
            "r_set", "r_spike", "r-pass", "r_winpoint",
            "l_winpoint", "l-pass", "l-spike", "l_set"
        ]
        self.person_activity_labels = {cls: i for i, cls in enumerate(person_activity_classes)}
        self.group_activity_labels = {cls: i for i, cls in enumerate(group_activity_classes)}
        
        # Initialize a dictionary to count how many samples exist per class
        self.person_activity_labels_count = {cls: 0 for cls in person_activity_classes}
        self.group_activity_labels_count = {cls: 0 for cls in group_activity_classes}
        
        
        # List of match IDs to include in this dataset
        self.matches = matches
        
        # List to hold all sample dicts with frame paths and labels
        self.samples = []
        self.get_samples()

    def get_samples(self):
        """
        Populate self.samples by iterating through specified matches and clips.
        Each sample includes the file path to the frame image and its category label.
        Also updates self.labels_count with class frequency statistics.
        """
        for match_id in self.matches:
            match_dct = self.annots[str(match_id)]
            
            for clip_id, clip_dct in match_dct.items():
                frames_category = clip_dct['category']
                self.group_activity_labels_count[frames_category] += 1
                frames_pathes = []
                boxes = []
                # For each frame in the clip, add sample info and update class count
                for frame_id, players in clip_dct['frames_boxes_dct'].items():
                    # Build absolute path to frame image file
                    frame_path = os.path.join(self.video_path, str(match_id), clip_id, f'{frame_id}.jpg')
                    frames_pathes.append(frame_path)
                    boxes.append(players)
                    for player in players :
                        self.person_activity_labels_count[player.category] += 1
                    
                self.samples.append({
                    'frames_pathes' : frames_pathes,
                    'frames_boxes' : boxes,
                    'clip_category' : frames_category
                })

    def __len__(self):
        """
        Returns the total number of samples (frames) in the dataset.
        """
        return len(self.samples)

    def _calc_center_box(self, x_min, x_max):
        x_center = (x_min + x_max) // 2
        return int(x_center),
        
    
    def extract_crops(self, frame, boxes = []):
        h, w = frame.shape[:2]
        crops = []
        boxes_centers = []
        
        categories = []
        
        for box in boxes :
            categories.append( torch.tensor(self.person_activity_labels[box.category] ,dtype=torch.long ))
            x_min, y_min, x_max, y_max = box.xMin, box.yMin, box.xMax, box.yMax
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h))
            if x_max <= x_min or y_max <= y_min:
                continue
            crop = frame[y_min:y_max, x_min:x_max]
            if self.transform:
                crop = self.transform(image=crop)["image"]   # 3 x 244 x 244

            box_center = self._calc_center_box(x_min,  x_max)
            crops.append((crop, box_center)) 
            


        crops = [ crops_ for crops_, box_center in sorted(crops, key = lambda k : k[1]) ]        
        
        while len(crops) < 12 :
            crops.append( torch.zeros((crops[0].size(0), crops[0].size(1), crops[0].size(2))) )
            categories.append(torch.tensor(-100, dtype=torch.long))

            
        if len(crops) > 12 :
            crops = crops[:12]
            categories = categories[:12]

        return torch.stack(crops) , torch.stack(categories)

    def __getitem__(self, idx):
        """
        Retrieve a sample at the specified index:
        - Loads the image from disk (as a NumPy array).
        - Converts BGR (OpenCV default) to RGB.
        - Applies Albumentations transforms (which convert to a tensor).
        - Returns:
            frame: torch.Tensor of shape (C, H, W), typically (3, 224, 224) if your transform resizes to 224x224.
            label: torch.Tensor scalar with dtype=torch.long, representing the class index.
        """
        sample = self.samples[idx]
        crops = []
        players_categories = []
        for frame_path, players in zip( sample['frames_pathes'], sample['frames_boxes'] ):

            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _crops, _players_categories = self.extract_crops(frame, players)
            crops.append(_crops)                        # 9 len with 12 x 3 x 244 x 244

            players_categories.append(_players_categories) # 9 len with 12 x 1

        crops = torch.stack(crops)                                               # 9 , 12 , 3 ,244 ,244
        players_categories = torch.stack(players_categories)                     # 9 , 12
        players_categories = players_categories.permute(1, 0).contiguous()       # 12 , 9
        players_categories = players_categories[:,-1]              # 12
                                                                        # If not seq like B6_A --> # 12,9
            
        label = torch.tensor(self.group_activity_labels[sample['clip_category']], dtype=torch.long) 

        
        return crops, players_categories, label   # 9,12,3,244,244     12,1          1