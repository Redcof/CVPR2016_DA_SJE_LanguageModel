import os
import pathlib
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from util.text_encoder_interface import CharEmbedTransform


class MultimodalDataset(Dataset):
    """
    A multimodal dataset wrapper. The directory must contain the following subdirectories and files:
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
                 images_dir="JPEGImages", caption_dir="captions", image_extension=".jpg"):
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
        self.image_files = [os.path.join(self.images_dir, file) for file in os.listdir(self.images_dir) if
                            file.endswith(image_extension)]
        # read captions
        self.caption_files = [os.path.join(self.caption_dir, file) for file in os.listdir(self.caption_dir) if
                              file.endswith(".txt")]
        self.labels = []
        with open(self.label_file) as fp:
            self.labels = {k: v for k, v in map(lambda x: x.strip().split(','), fp.readlines())}
        assert len(self.image_files) == len(
            self.caption_files), "The count of images and captions must be same, got {}!={}".format(len(self.image_files),
                                                                                                    len(self.caption_files))
        if image_transform is None:
            image_transform = transforms.Compose([
                transforms.Resize(img_dim),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        if caption_transform is None:
            char_embedding = CharEmbedTransform(self.doc_length)
            self.vocab_length = char_embedding.vocab_length
            caption_transform = transforms.Compose([
                char_embedding
            ])
        
        if target_transform is None:
            target_transform = transforms.Compose([
                transforms.Lambda(self._class_encoder),
            ])
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.target_transform = target_transform
        self.class_ids = [cls_idx for cls_idx, lbl in enumerate(self.classnames)]
        self.dtype = torch.float16
    
    def _class_encoder(self, label):
        try:
            return torch.tensor(self.classname_dict[label])
        except:
            return torch.tensor(-1)
    
    # def _basic_char_encoder(self, caption):
    #     """Must remove '\n' character. It currently supports ASCII characters"""
    #     # lowercase
    #     caption = caption.lower().strip()
    #
    #     def __get_index(word, space=True):
    #         if space:
    #             yield self.vocabulary.index(' ')
    #         for char in list(word):
    #             try:
    #                 yield self.vocabulary.index(char)
    #             except:
    #                 yield 0
    #
    #     char_ids = []
    #     # encode characters
    #     [char_ids.extend(list(__get_index(list(char)))) for char in caption.split()]
    #     char_ids = char_ids[1:]  # skip the first character as it is always 'space'
    #
    #     # truncate if the sentence is large
    #     if len(char_ids) > self.doc_length:
    #         char_ids = char_ids[:self.doc_length]
    #
    #     # place encoding into a tensor
    #     char_encode = torch.zeros(self.doc_length, self.vocab_length)
    #     for idx, char_code in enumerate(char_ids[1:]):
    #         try:
    #             char_encode[idx, char_code] = 1
    #         except IndexError as e:
    #             print("Error: CaptionL:", caption, len(caption), idx, char_code, len(char_ids), char_encode.shape)
    #             raise e
    #     return char_encode
    #
    def get_img(self, img_path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img = self.image_transform(img)
        return img
    
    def get_caption_embedding(self, caption_path, ) -> str:
        with open(caption_path) as fp:
            captions = fp.readlines()
            captions_ix = random.randint(0, len(captions) - 1)
            caption = captions[captions_ix]
            caption = self.caption_transform(caption)
            return caption
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        caption_file = self.caption_files[idx]
        image = self.get_img(image_file)
        caption = self.get_caption_embedding(caption_file)
        filename = os.path.basename(image_file)
        label = self.target_transform(self.labels[filename])
        return caption, image, label
    
    def __len__(self):
        return len(self.image_files)
