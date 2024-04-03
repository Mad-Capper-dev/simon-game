#Register dataset as torchvision CocoDetection
import os

from dataset import train_dataset, val_dataset, show_random_element
from image_helper import COLORS, rescale_bboxes, visualize_predictions
from camera_helper import CameraHelper
from model import Model
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


# Create the model
model.create()

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

# We can now manually check our model using the webcam
CameraHelper(model, show=False).captureAndPredict()