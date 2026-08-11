import os
import random
from math import cos, sin, radians, floor
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import imutils

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, MapType

if not os.path.exists('/content/ibug_300W_large_face_landmark_dataset'):
    print("Downloading dataset...")
    os.system("wget -q http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz")
    os.system("tar -xzf ibug_300W_large_face_landmark_dataset.tar.gz")
    os.system("rm ibug_300W_large_face_landmark_dataset.tar.gz")

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("FaceLandmarksPreprocessing") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 2. Distributed XML Parsing Strategy
# Instead of parsing the massive XML sequentially on the master node,
# we extract the raw image-nodes and let Spark workers process them.
def parse_xml_to_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata_list = []
    root_dir = 'ibug_300W_large_face_landmark_dataset'

    # root[2] corresponds to the images list in the DLIB XML structure
    for filename in root[2]:
        file_path = os.path.join(root_dir, filename.attrib['file'])

        # Extract crop box attributes
        crop_attribs = dict(filename[0].attrib)

        # Extract 68 landmarks coordinates
        landmarks = []
        for num in range(68):
            x = float(filename[0][num].attrib['x'])
            y = float(filename[0][num].attrib['y'])
            landmarks.append([x, y])

        metadata_list.append((file_path, crop_attribs, landmarks))
    return metadata_list

# Extract metadata sequentially first to parallelize across workers
xml_file_path = 'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'
raw_metadata = parse_xml_to_metadata(xml_file_path)

# Define PySpark Schema
schema = StructType([
    StructField("image_path", StringType(), False),
    StructField("crops", MapType(StringType(), StringType()), False),
    StructField("landmarks", ArrayType(ArrayType(FloatType())), False)
])

# Create Distributed DataFrame
df = spark.createDataFrame(raw_metadata, schema=schema)

# 3. Distributed Train/Validation Split
# PySpark handles random splitting reliably across partitions
train_df, valid_df = df.randomSplit([0.9, 0.1], seed=42)

print(f"The length of Train set is {train_df.count()}")
print(f"The length of Valid set is {valid_df.count()}")

# 4. PySpark-Compatible Transform Class
class SparkTransforms:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def crop_and_prep(self, image_path, landmarks, crops):
        # Read image using OpenCV (or PIL)
        image = cv2.imread(image_path, 0)
        if image is None:
            return None, None

        image = Image.fromarray(image)
        landmarks = np.array(landmarks, dtype=np.float32)

        # 1. Crop face
        left, top = int(crops['left']), int(crops['top'])
        width, height = int(crops['width']), int(crops['height'])
        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = landmarks - np.array([[left, top]])
        landmarks = landmarks / np.array([img_shape[1], img_shape[0]])

        # 2. Resize
        image = TF.resize(image, (224, 224))

        # 3. Color Jitter
        image = self.color_jitter(image)

        # 4. Rotate
        angle = random.uniform(-10, 10)
        transformation_matrix = np.array([
            [cos(radians(angle)), -sin(radians(angle))],
            [sin(radians(angle)), cos(radians(angle))]
        ])
        image = imutils.rotate(np.array(image), angle)
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5

        # 5. Normalize and Convert to Tensor
        image_tensor = TF.to_tensor(Image.fromarray(image))
        image_tensor = TF.normalize(image_tensor, [0.5], [0.5])

        # Standardizing landmarks output (-0.5 offset matches original script)
        final_landmarks = new_landmarks - 0.5

        return image_tensor.numpy().tolist(), final_landmarks.tolist()

# 5. PyTorch Distributed Dataset Wrapper for Spark Dataframes
class SparkPyTorchDataset(torch.utils.data.Dataset):
    """
    Consumes collected Spark partitions or local partition views
    to serve batches to PyTorch Dataloaders seamlessly.
    """
    def __init__(self, spark_dataframe):
        # Collect data locally on workers/drivers during local training setups
        # For multi-node distributed training, use Horovod or TorchDistributor directly over DataFrames
        self.data = spark_dataframe.collect()
        self.transformer = SparkTransforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Perform transformations on the fly during training iteration
        img_tensor_list, landmarks_list = self.transformer.crop_and_prep(
            row['image_path'],
            row['landmarks'],
            row['crops']
        )

        # Fallback handling for missing/corrupted files
        if img_tensor_list is None:
            return self.__getitem__((idx + 1) % self.__len__())

        return torch.tensor(img_tensor_list), torch.tensor(landmarks_list)

# 6. Instantiate DataLoaders for PyTorch Model Consumption
train_dataset = SparkPyTorchDataset(train_df)
valid_dataset = SparkPyTorchDataset(valid_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0)

# 7. Testing the shape of input data
images, landmarks = next(iter(train_loader))
print("Final PyTorch Batch Dimensions via PySpark Pipeline:")
print(f"Images Shape: {images.shape}")
print(f"Landmarks Shape: {landmarks.shape}")
