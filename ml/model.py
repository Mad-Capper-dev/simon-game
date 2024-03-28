
import pytorch_lightning as pl
from dataset import CocoDetection, feature_extractor
from transformers import DetrConfig, AutoModelForObjectDetection
import torch
import os
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['labels'] = labels
  return batch

class Detr(pl.LightningModule):
     

     def __init__(self, train_dataset, val_dataset, lr, weight_decay, id2label):
         super().__init__()
         self.train_dataset = train_dataset
         self.val_dataset = val_dataset
         # replace COCO classification head with custom head
         self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.weight_decay = weight_decay
         self.id2label = id2label

     def forward(self, pixel_values):
       outputs = self.model(pixel_values=pixel_values)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                  weight_decay=self.weight_decay)
        return optimizer
     
     def load(self, version, checkpoint_name):
        script_dir = os.path.dirname(__file__)
        return Detr.load_from_checkpoint(os.path.join(script_dir, f"../lightning_logs/version_{version}/checkpoints/{checkpoint_name}.ckpt"), train_dataset=self.train_dataset, val_dataset=self.val_dataset, lr=2.5e-5, weight_decay=1e-4, id2label=self.id2label)

     def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)

     def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=collate_fn, batch_size=1)
     
     def batch(self):
        return next(iter(self.train_dataloader()))


class Model():
   train_dataset: CocoDetection
   val_dataset: CocoDetection
   train_dataloader: DataLoader
   val_dataloader: DataLoader
   id2label: dict
   batch: any

   detr: Detr

   def __init__(self, train_dataset, val_dataset):
      self.train_dataset = train_dataset
      self.val_dataset = val_dataset
      self.train_dataloader = DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
      self.val_dataloader = DataLoader(self.val_dataset, collate_fn=collate_fn, batch_size=1)
      self.batch = next(iter(self.train_dataloader))
      self.id2label = {k: v['name'] for k,v in train_dataset.coco.cats.items()}

   def create(self):
      self.detr = Detr(train_dataset=self.train_dataset, val_dataset=self.val_dataset, lr=2.5e-5, weight_decay=1e-4, id2label=self.id2label)
      return self.detr

   def train(self):
     
      trainer = Trainer( max_steps=2000, gradient_clip_val=0.1, accumulate_grad_batches=4)
      trainer.fit(self.detr)

      return self.detr 

   def load(self, version, checkpoint_name):
      self.detr = Detr(train_dataset=self.train_dataset, val_dataset=self.val_dataset, lr=2.5e-5, weight_decay=1e-4, id2label=self.id2label).load(version, checkpoint_name)
      return self.detr

   def output(self, pixel_values ):
      return self.detr(pixel_values=pixel_values)