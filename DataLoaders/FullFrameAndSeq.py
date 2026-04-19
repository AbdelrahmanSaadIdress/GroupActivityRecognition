import os 
import cv2
import torch
from torch.utils.data import Dataset

class FullFrameAndSeq(Dataset):
    def __init__(self, annots, frames_annots_path, transform, matches, seq = True):
        """
        Args:
            annots (dict): Dictionary containing annotations organized by match and clip.
            frames_annots_path (str): Root path where frame images are stored.
            transform (albumentations.Compose): Albumentations transform pipeline to apply on frames.
            matches (list): List of match IDs (as strings or integers) to include in this dataset.
        """
        self.annots = annots 
        self.video_path = frames_annots_path
        self.transform = transform
        self.seq = seq 
        
        # Define the fixed set of group activity classes and map them to numeric labels
        group_activity_classes = [
            "r_set", "r_spike", "r-pass", "r_winpoint",
            "l_winpoint", "l-pass", "l-spike", "l_set"
        ]
        self.labels = {cls: i for i, cls in enumerate(group_activity_classes)}
        
        # Initialize a dictionary to count how many samples exist per class
        self.labels_count = {cls: 0 for cls in group_activity_classes}
        
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
                
                # For each frame in the clip, add sample info and update class count
                frames_pathes = []
                for frame_id in clip_dct['frames_boxes_dct']:
                    self.labels_count[frames_category] += 1
                    
                    # Build absolute path to frame image file
                    frame_path = os.path.join(self.video_path, str(match_id), clip_id, f'{frame_id}.jpg')

                    if not self.seq :
                        self.samples.append({
                            'frames_pathes': [frame_path],
                            'category': frames_category
                        })

                    if self.seq :
                        frames_pathes.append(frame_path)
                if self.seq :
                    self.samples.append({
                    'frames_pathes' : frames_pathes,
                    'category' : frames_category
                    })
                        

    def __len__(self):
        """
        Returns the total number of samples (frames) in the dataset.
        """
        return len(self.samples)

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
        frames = []
        for frame_path in sample['frames_pathes']:
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if self.transform:
                frame = self.transform(image=frame)["image"]  

            frames.append(frame)

        
            
        label = torch.tensor(self.labels[sample['category']], dtype=torch.long)  
    
        return torch.stack(frames).squeeze(0), label   # tensor with shape (3, H, W)   ,    scalar tensor    

# seq:
# Batch shape = torch.Size([4, 9, 3, 224, 224])  (B,T,C,H,W)
# Labels = tensor([4, 2, 6, 2])