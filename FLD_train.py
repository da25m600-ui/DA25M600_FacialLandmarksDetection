import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import *
import xml.etree.ElementTree as ET

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.pytorch
mlflow.set_experiment("Facial LandMarks Detection")

with mlflow.start_run(run_name="Best_Model") as run:
  class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x

  torch.autograd.set_detect_anomaly(True)
  network = Network()
  network.cuda()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(network.parameters(), lr=0.0001)

  loss_min = np.inf
  num_epochs = 5

  start_time = time.time()
  for epoch in range(1,num_epochs+1):
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    l_t = []
    l_v = []
    network.train()
    for step in range(1,len(train_loader)+1):
        images, landmarks = next(iter(train_loader))
        images = images.cuda()
        landmarks = landmarks.view(landmarks.size(0),-1).cuda()
        predictions = network(images)
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        # find the loss for the current step
        loss_train_step = criterion(predictions, landmarks)
        # calculate the gradients
        loss_train_step.backward()
        # update the parameters
        optimizer.step()
        loss_train += loss_train_step.item()
        running_loss = loss_train/step
        l_t.append(running_loss)
        #print_overwrite(step, len(train_loader), running_loss, 'train')
    print("Training Loss in epoch", epoch )
    plt.plot(range(1,len(train_loader)+1), l_t)
    plt.xlabel('Iteration Steps')
    plt.ylabel('Training Loss')
    plt.title('Training loss in a complete epoch')
    plt.show()
    network.eval()
    with torch.no_grad():
        for step in range(1,len(valid_loader)+1):
          images, landmarks = next(iter(valid_loader))
          images = images.cuda()
          landmarks = landmarks.view(landmarks.size(0),-1).cuda()
          predictions = network(images)
          # find the loss for the current step
          loss_valid_step = criterion(predictions, landmarks)
          loss_valid += loss_valid_step.item()
          running_loss = loss_valid/step
          l_v.append(running_loss)
          #print_overwrite(step, len(valid_loader), running_loss, 'valid')
    print("Validation Loss in epoch", epoch )
    plt.plot(range(1,len(valid_loader)+1), l_v)
    plt.xlabel('Iteration Steps')
    plt.ylabel('Validation loss')
    plt.title('Validation loss in a complete epoch')
    plt.show()
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')
    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), '/content/face_landmarks.pth')
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')
  print('Training Complete')
  print("Total Elapsed Time : {} s".format(time.time()-start_time))

  # Save, log, and register the optimal model instance inside the MLflow Model Registry
  model_info = mlflow.pytorch.log_model(
    pytorch_model=network,
    artifact_path="best_network",
    registered_model_name="Resnet_Facial_Landmarks_Model",
    serialization_format="pickle"  # <--- Add this line
    )
  mlflow.log_metrics({
            "MSE_Validation_loss": loss_min
        }, step=epoch)
  #mlflow.log_artifact(images, artifact_path="images")
  #mlflow.log_artifact(landmarks, artifact_path="landmarks")
  #mlflow.log_artifact(predictions, artifact_path="Validation_predictions")

print(f"Model successfully saved and registered at: {model_info.model_uri}")
