import torchvision
import os
import numpy as np
import os
from PIL import Image, ImageDraw
from transformers import AutoFeatureExtractor

class CocoDetection(torchvision.datasets.CocoDetection):
    """ Coco datasets helper"""
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny", size=512, max_size=864)

script_dir = os.path.dirname(__file__)
dataset_dir =  os.path.join(script_dir, "data")

train_dataset = CocoDetection(img_folder=os.path.join(dataset_dir, "train"), feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=os.path.join(dataset_dir, "valid"), feature_extractor=feature_extractor, train=False)

# Shows a random image from the training dataset and annotate it with the corresponding annotatio
def show_random_element(dataset: CocoDetection):
    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = dataset.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image n°{}'.format(image_id))
    image = dataset.coco.loadImgs(image_id)[0]
    image_path = os.path.join(os.path.join(dataset_dir, "train"), image['file_name'])
    print("Image path : " + image_path)
    image = Image.open(image_path)

    annotations = dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')

    image.show()