import numpy as np
import cv2
import os
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        labels, 
        tfms, 
        #feature_extractor,
        tfms_orig=None,
        input_shape=[320, 320],
        image_modification = ""
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.tfms = tfms
        self.tfms_orig = tfms_orig
        #self.feature_extractor = feature_extractor
        self.input_shape = input_shape
        self.image_modification=image_modification
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        # Check if the file exists
        if not os.path.exists(self.image_paths[index]):
            raise FileNotFoundError(f"Image file not found: {self.image_paths[index]}")

        # Load original image
        try:
            orig_image = cv2.imdecode(np.fromfile(self.image_paths[index], dtype=np.uint8), cv2.IMREAD_UNCHANGED).astype('uint8')
        except:
            raise ValueError(f"Failed to read image: {self.image_paths[index]}")

        # Turn black and white images on one channel to 3 channels (RGB)
        if len(orig_image.shape)==2:
            orig_image = np.stack((orig_image,)*3, axis=-1)
            
            if image_modification == "Histogram_equalization":
                orig_image = cv2.cvtColor(orig_image, cv.COLOR_BGR2GRAY)
                orig_image = cv2.equalizeHist(orig_image)

                if len(orig_image.shape)==2:
                    orig_image = np.stack((orig_image,)*3, axis=-1)

        # Apply transforms
        if self.tfms is not None:
            transformed = self.tfms(image=orig_image)
            image = transformed['image']
        else:
            image = orig_image
        
        # Convert to C, H, W
        image = image.transpose(2, 0, 1)

        #print("image.shape: "+str(image.shape))

        # Apply transforms to original image if specified
        if self.tfms_orig is not None:
            transformed = self.tfms_orig(image=orig_image)
            orig_image = transformed['image']

        label = self.labels[index]

        return image, label, orig_image

# Example usage:
# image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
# labels = [0, 1, 0, ...]  # Corresponding labels for each image
# tfms = some_transforms  # Your image transformations
# feature_extractor = some_feature_extractor  # Your feature extractor
# dataset = ClassificationDataset(image_paths, labels, tfms, feature_extractor)
