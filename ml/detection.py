#Register dataset as torchvision CocoDetection
import torchvision
import os
from transformers import AutoFeatureExtractor

from dataset import CocoDetection, feature_extractor, train_dataset, val_dataset, show_random_element
from image_helper import COLORS, rescale_bboxes, visualize_predictions
from camera_helper import CameraHelper
from model import Detr, Model
from PIL import Image


script_dir = os.path.dirname(__file__)
dataset_dir =  os.path.join(script_dir, "data")

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

# Verifies loading of training dataset
show_random_element(train_dataset)

# Verifies loading of validation dataset
# show_random_element(val_dataset)

model = Model(train_dataset=train_dataset, val_dataset=val_dataset)


# Load Model or create it
model.load( version=0, checkpoint_name="epoch=11-step=456")
# model.create()

# Train it 
model.train()

#We verify our model on an element of the validation dataset
pixel_values, target = val_dataset[1]
pixel_values = pixel_values.unsqueeze(0)
outputs = model.output(pixel_values=pixel_values)
     
image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(os.path.join(dataset_dir, "valid"), image['file_name']))

visualize_predictions(image, outputs)

CameraHelper(model).captureAndPredict()