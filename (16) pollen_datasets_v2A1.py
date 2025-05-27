# "pollen_datasets_v2A1.py" ~ "v2.151"
# Enhanced version of the PollenDataset class with better error handling and memory efficiency.
# Based on v2A with improvements from v2B and v2E.

import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms

# Setup a simple logger for this module
log_formatter = logging.Formatter('%(asctime)s [Dataset v2A1] %(levelname)s - %(message)s')
logger = logging.getLogger("PollenDatasetLogger_v2A1")
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if this module is reloaded
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

class PollenDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed 128x128 greyscale pollen crops.
    Handles resizing on-the-fly if needed (though preprocessing is recommended).
    Converts images to 1-channel greyscale tensor.
    Applies specified transformations.
    """
    def __init__(self, root_dir, image_size=128, channels_img=1, transform=None):
        """
        Args:
            root_dir (str): Directory with all the preprocessed image files.
            image_size (int): Target size (height and width) for the images.
            channels_img (int): Target number of channels (should be 1).
            transform (callable, optional): Optional transform to be applied on a sample.
                                            (Should include ToTensor and Normalize).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.channels_img = channels_img

        if self.transform is None:
            logger.warning("No transform provided to PollenDataset. Using default ToTensor + Normalize.")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5] * self.channels_img, [0.5] * self.channels_img)
            ])

        self.image_files = []
        if not os.path.isdir(root_dir):
             logger.error(f"Input directory not found: {root_dir}")
             raise FileNotFoundError(f"Input directory not found: {root_dir}")

        logger.info(f"Scanning for images in: {root_dir}")
        for root, _, files in os.walk(root_dir):
            for f in files:
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                      self.image_files.append(os.path.join(root, f))

        if not self.image_files:
             logger.warning(f"No images found in {root_dir}. Check path and file extensions.")
        else:
             logger.info(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
             logger.error(f"Index {idx} out of bounds for image_files list (len: {len(self.image_files)})")
             return self._get_dummy_tensor()

        img_path = self.image_files[idx]
        try:
            # Open image and ensure greyscale ('L' mode)
            image = Image.open(img_path).convert('L')

            # Basic check if image size matches target
            if image.size != (self.image_size, self.image_size):
                 # If preprocessing failed or wasn't run, resize here
                 image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            # Ensure output tensor has the correct number of channels
            if image.shape[0] != self.channels_img:
                 logger.warning(f"Image tensor shape {image.shape} mismatch channels {self.channels_img} for {img_path}. Fixing.")
                 if image.shape[0] > self.channels_img:
                     image = image[0:self.channels_img, :, :]
                 elif image.shape[0] == 1 and self.channels_img > 1:
                     image = image.repeat(self.channels_img, 1, 1)
                 else:
                     return self._get_dummy_tensor()
            
            return image
            
        except (UnidentifiedImageError, OSError) as e:
             logger.warning(f"Skipping corrupted/unreadable image: {img_path} ({e})")
             return self._get_dummy_tensor()
        except Exception as e:
             logger.error(f"Unexpected error loading image {img_path}: {e}", exc_info=True)
             return self._get_dummy_tensor()

    def _get_dummy_tensor(self):
        """Creates a dummy tensor normalized like others."""
        dummy_tensor = torch.zeros((self.channels_img, self.image_size, self.image_size))
        if isinstance(self.transform, transforms.Compose):
            for t in self.transform.transforms:
                 if isinstance(t, transforms.Normalize):
                      try:
                          dummy_tensor = t(dummy_tensor.float())
                      except Exception as norm_e:
                          logger.error(f"Failed to normalize dummy tensor: {norm_e}")
                      break
        return dummy_tensor
