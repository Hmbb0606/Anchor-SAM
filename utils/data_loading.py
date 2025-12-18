import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BasicDataset(Dataset):
    """ Basic dataset for train, evaluation and test.

    Attributes:
        images_dir(str): path of images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        ids(list): name list of images.
        transforms(class): data augmentation pipeline applied to image and label.

    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        """ Init of basic dataset.

        Parameter:
            images_dir(str): path of images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.

        """

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if self.train:
            self.transforms = A.Compose([
                A.Resize(height=1024, width=1024, interpolation=cv2.INTER_LINEAR, always_apply=True),

                A.HueSaturationValue(
                    hue_shift_limit=(-30, 30),
                    sat_shift_limit=(-5, 5),
                    val_shift_limit=(-15, 15),
                    p=0.5
                ),

                A.ShiftScaleRotate(
                    shift_limit=(-0.1, 0.1),
                    scale_limit=(-0.1, 0.1),
                    rotate_limit=(0, 0),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Lambda(image=lambda img, **kwargs: (img / 255.0 * 3.2 - 1.6).astype(np.float32)),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(height=1024, width=1024, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Lambda(image=lambda img, **kwargs: (img / 255.0 * 3.2 - 1.6).astype(np.float32)),
                ToTensorV2(),
            ])

    def __len__(self):
        """ Return length of dataset."""
        return len(self.ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""
        if len(label.shape) == 3 and label.shape[2] in [3, 4]:
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

        # 将掩码二值化为 0 和 1
        label[label != 0] = 1
        return label

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""
        img = Image.open(filename)
        img = np.array(img).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        """ Index dataset.

        Index image name list to get image name, search image in image path with its name,
        open image and convert it to array.

        Preprocess array, apply data augmentation on it, and convert array to tensor.

        Parameter:
            idx(int): index of dataset.

        Return:
            tensor(tensor): tensor of image.
            label_tensor(tensor): tensor of label.
            name(str): the same name of image and label.
        """

        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        label_file = list(self.labels_dir.glob(name + '.*'))

        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {name}: {label_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        img = self.load(img_file[0])
        label = self.load(label_file[0])

        label = self.label_preprocess(label)

        sample = self.transforms(image=img, mask=label)
        tensor, label_tensor = sample['image'], sample['mask']

        tensor = tensor.contiguous()
        label_tensor = label_tensor.contiguous()

        return tensor, label_tensor, name

