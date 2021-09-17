import json
import tqdm
from collections import OrderedDict
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './epoch_0_checkpoint.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
normalize_thermal = transforms.Normalize(mean=[0.485],std=[0.229])



def detect(prediction_json, index, rgb_image, thermal_image, min_score, max_overlap, top_k, suppress=None):

    rgb = normalize(to_tensor(resize(rgb_image)))
    rgb = rgb.to(device)  
    thermal = normalize_thermal(to_tensor(resize(thermal_image)))
    thermal = thermal.to(device) 

    predicted_locs, predicted_scores = model(rgb.unsqueeze(0),thermal.unsqueeze(0)) # Forward prop.

    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k) # Detect objects in SSD output
    det_boxes = det_boxes[0].to('cpu') # Move detections to the CPU

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([rgb_image.width, rgb_image.height, rgb_image.width, rgb_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    
    det_boxes=det_boxes.tolist()
    det_labels=det_labels[0].tolist()
    det_scores=det_scores[0].tolist()
    
    for j in range(len(det_boxes)):
      det_boxes[j][2]=det_boxes[j][2]-det_boxes[j][0]
      det_boxes[j][3]=det_boxes[j][3]-det_boxes[j][1]

    for idx in range(len(det_labels)):
      prediction=OrderedDict()
      prediction["image_id"]=index
      prediction["category_id"]=det_labels[idx]
      prediction["bbox"]=det_boxes[idx]
      prediction["score"]=det_scores[idx]

      prediction_json.append(prediction)

if __name__ == '__main__':

  pd_json=list()

  with open('../Json/TEST_RGB.json', 'r') as rgb:
    rgb_images = json.load(rgb)

  with open('../Json/TEST_THERMAL.json', 'r') as th:
    thermal_images = json.load(th)


  for i in tqdm.tqdm(range(len(rgb_images))):

    rgb_image = Image.open(rgb_images[i], mode='r')
    rgb_image = rgb_image.convert('RGB')

    thermal_image = Image.open(thermal_images[i], mode='r')
    thermal_image = thermal_image.convert('L')

    detect(pd_json, i, rgb_image, thermal_image, min_score=0.2, max_overlap=0.5, top_k=200)
  
  with open('../Json/prediction.json','w',encoding="utf-8") as n:
    json.dump(pd_json,n,ensure_ascii=False,indent="\t")
  
