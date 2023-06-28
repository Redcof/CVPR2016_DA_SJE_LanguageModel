import csv
import os
import pathlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import PIL
from torchvision.transforms.transforms import _setup_size


class AspectResize(torch.nn.Module):
    """
   Resize image while keeping the aspect ratio.
   Extra parts will be covered with 255(white) color value
   """
    
    def __init__(self, size, background=255):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.background = background
    
    @staticmethod
    def fit_image_to_canvas(image: Image, canvas_width, canvas_height, background=255) -> Image:
        # Get the dimensions of the image
        image_width, image_height = image.size
        
        # Calculate the aspect ratio of the image
        image_aspect_ratio = image_width / float(image_height)
        
        # Calculate the aspect ratio of the canvas
        canvas_aspect_ratio = canvas_width / float(canvas_height)
        
        # Calculate the new dimensions of the image to fit the canvas
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = canvas_height * image_aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / image_aspect_ratio
        
        # Resize the image to the new dimensions
        image = image.resize((int(new_width), int(new_height)), PIL.Image.BICUBIC)
        
        # Create a blank canvas of the specified size
        canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
        canvas[:, :, :] = background
        
        # Calculate the position to paste the resized image on the canvas
        x = int((canvas_width - new_width) / 2)
        y = int((canvas_height - new_height) / 2)
        
        # Paste the resized image onto the canvas
        canvas[y:y + int(new_height), x:x + int(new_width)] = np.array(image)
        
        return PIL.Image.fromarray(canvas)
    
    def forward(self, image: Image) -> Image:
        image = self.fit_image_to_canvas(image, self.size[0], self.size[1], self.background)
        return image


class MultimodalDataset(Dataset):
    """
    A multimodal dataset wrapper. The directory must contain the following sub-directories and files:
    data_root
        | - train
        |      | - JPEGImages
        |      |       | -  FILE01.jpeg
        |      |       | -  FILE02.jpeg
        |      |       | -  FILE03.jpeg
        |      | - Captions
        |              | -  FILE01.txt
        |              | -  FILE02.txt
        |              | -  FILE03.txt
        | - classes.txt
    """
    
    def __init__(self, dataset_path, classnames, label_file, doc_length, img_dim,
                 image_transform=None, caption_transform=None, target_transform=None,
                 split="train",
                 images_dir="JPEGImages", caption_dir="Captions", image_extension=".jpg"):
        assert split in ("train", "text")
        dataset_path = pathlib.Path(dataset_path)
        self.dataset_path = dataset_path
        self.images_dir = dataset_path / split / images_dir
        self.caption_dir = dataset_path / split / caption_dir
        self.label_file = pathlib.Path(label_file)
        # read class names
        self.classnames = classnames
        self.classname_dict = {k: v for v, k in enumerate(classnames)}
        assert len(self.classnames) > 0, "A list of classnames is required"
        self.doc_length = doc_length
        # read images
        self.images = [os.path.join(self.images_dir, file) for file in os.listdir(self.images_dir) if
                       file.endswith(image_extension)]
        # read captions
        self.captions = [os.path.join(self.caption_dir, file) for file in os.listdir(self.caption_dir) if
                         file.endswith(".txt")]
        self.labels = []
        with open(self.label_file) as fp:
            self.labels = {k: v for k, v in map(lambda x: x.strip().split(','), fp.readlines())}
        assert len(self.images) == len(
            self.captions), "The count of images and captions must be same, got {}!={}".format(len(self.images),
                                                                                               len(self.captions))
        if image_transform is None:
            image_transform = transforms.Compose([
                transforms.Resize(img_dim),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        if caption_transform is None:
            caption_transform = transforms.Compose([
                transforms.Lambda(self._basic_char_encoder),
            ])
        
        if target_transform is None:
            target_transform = transforms.Compose([
                transforms.Lambda(self._class_encoder),
            ])
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.target_transform = target_transform
        self.vocabulary = "\nabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.vocab_length = len(self.vocabulary)
        self.class_ids = [cls_idx for cls_idx, lbl in enumerate(self.classnames)]
        self.dtype = torch.float16
    
    def _class_encoder(self, label):
        try:
            return torch.tensor(self.classname_dict[label])
        except:
            return torch.tensor(-1)
    
    def _basic_char_encoder(self, caption):
        """Must remove '\n' character. It currently supports ASCII characters"""
        # lowercase
        caption = caption.lower().strip()
        
        def __get_index(word, space=True):
            if space:
                yield self.vocabulary.index(' ')
            for char in list(word):
                try:
                    yield self.vocabulary.index(char)
                except:
                    yield 0
        
        char_ids = []
        # encode characters
        [char_ids.extend(list(__get_index(list(char)))) for char in caption.split()]
        char_ids = char_ids[1:]  # skip the first character as it is always 'space'
        
        # truncate if the sentence is large
        if len(char_ids) > self.doc_length:
            char_ids = char_ids[:self.doc_length]
        
        # place encoding into a tensor
        char_encode = torch.zeros(self.doc_length, self.vocab_length)
        for idx, char_code in enumerate(char_ids[1:]):
            try:
                char_encode[idx, char_code] = 1
            except IndexError as e:
                print("Error: CaptionL:", caption, len(caption), idx, char_code, len(char_ids), char_encode.shape)
                raise e
        return char_encode
    
    def get_img(self, img_path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img = self.image_transform(img)
        return img
    
    def get_caption(self, caption_path, ) -> str:
        with open(caption_path) as fp:
            captions = fp.readlines()
            captions_ix = random.randint(0, len(captions) - 1)
            caption = captions[captions_ix]
            caption = self.caption_transform(caption)
            return caption
    
    def __getitem__(self, idx):
        image_file = self.images[idx]
        caption_file = self.captions[idx]
        image = self.get_img(image_file)
        caption = self.get_caption(caption_file)
        filename = os.path.basename(image_file)
        label = self.target_transform(self.labels[filename])
        return caption, image, label
    
    def __len__(self):
        return len(self.images)
