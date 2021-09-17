import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from utils import transform


class KAISTDataset(Dataset):

    def __init__(self,split):
    
      self.split=split
      self.thermal=[]
      self.rgb=[]
      self.objects=[]

      with open('../Json/TRAIN_RGB.json', 'r') as r:
        self.rgb = json.load(r)

      with open('../Json/TRAIN_THERMAL.json', 'r') as t:
        self.thermal = json.load(t)

      with open('../Json/TRAIN_objects.json', 'r') as o:
        self.objects = json.load(o)
      
      assert len(self.rgb) == len(self.objects) == len(self.thermal)

    def __getitem__(self, i):
        
        rgb_image = Image.open(self.rgb[i], mode='r')
        rgb_image = rgb_image.convert('RGB') 

        thermal_image = Image.open(self.thermal[i], mode='r')
        thermal_image = thermal_image.convert('L') 

        objects = self.objects[i]

        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)
    
        # Apply transformations
        rgb_image, thermal_image, boxes, labels, difficulties = transform(rgb_image, thermal_image, boxes, labels, difficulties, split=self.split)

        return rgb_image, thermal_image, boxes, labels, difficulties

    def __len__(self):
        return len(self.rgb)

    def collate_fn(self, batch):
        
        rgb = list()
        thermal=list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            rgb.append(b[0])
            thermal.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])
            
        rgb = torch.stack(rgb, dim=0)
        thermal = torch.stack(thermal, dim=0)

        return rgb, thermal, boxes, labels, difficulties