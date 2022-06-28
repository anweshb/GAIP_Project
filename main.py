import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import timm
import warnings
import pyttsx3
import sys
from PIL import Image
warnings.filterwarnings("ignore")

custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/best.pt', force_reload = False)
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type, force_reload = False)

custom_model.names = ['NULL','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses' ,'handbag', 'tie', 
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard','tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk' ,'toilet', 'door', 'tv', 'laptop',
         'mouse', 'remote', 'keyboard', 'cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator','blender','book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush', 'hair brush']

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


transform = midas_transforms.small_transform


filename = sys.argv[1]

#show_img = plt.imread(filename)
#plt.imshow(show_img)
#plt.show()


im = Image.open(filename)
im.show()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
#midas.eval()

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
plt.imshow(output)
plt.show()
plt.savefig('heatmap.png')

im = Image.open('heatmap.png')
im.show()

width = output.shape[1]
height = output.shape[0]

new_output = output[:int(height*0.7),:]
#plt.imshow(new_output)

def split_depthmap(depthmap):
    
    width = depthmap.shape[1]
    height = depthmap.shape[0]
    
    depthmap = depthmap[:int(height*0.7),:]
    
    max_depth_point = (np.where(depthmap== np.max(depthmap)))[1][0]
    
    if max_depth_point <= (width//3):
        return "left"
    elif (width//3) < max_depth_point <= (width * 2/3):
        return "middle"
    else:
        return "right"

def split_image(image, side):
    
    width = image.shape[1]
    height = image.shape[0]
    
    left_side = image[:,:width//3,:]
    middle = image[:,width//3:2*(width//3),:]
    right_side = image[:,2*(width//3):,:]
    
    if side == "left":
        return left_side
    elif side == "middle":
        return middle
    else:
        return right_side

def give_alert(img, depthmap):
    
    aoi_region = split_depthmap(depthmap)     ###WHICH SIDE OBJECT IS ON
    
    aoi_image = split_image(img, aoi_region)
    
    results = custom_model(aoi_image)
    
    if results.pandas().xyxy[0].empty:
        ##GIVE AUDIO ALERT FOR UNKNOWN OBJECT at detected side
        to_return = "Unknown object at {} side".format(aoi_region) 
        return to_return
    else:
        obj = results.pandas().xyxy[0].iloc[0]['name']
        ## GIVE AUDIO ALERT FOR obj and side
        print(f"{obj} detected at your {aoi_region}")
        
        to_return = "{} detected at {} side".format(obj, aoi_region)
        return to_return

def pipeline(img_path):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img[:]

    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    
    return give_alert(img, output)

#pipeline(filename)
txt = pipeline(filename)

#print(txt)



# initialisation
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-50)
engine.say(txt)
engine.runAndWait()
