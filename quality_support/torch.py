import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class GrabberDataFrameDataset(Dataset):
    def __init__(self, grabber, dataframe: pd.DataFrame, transform=None):
        super().__init__()
        self.data_hashes = dataframe["data_hash"]
        self.object_hashes = dataframe["object_hash"]
        self.class_labels = dataframe["class_labels"]
        self.transform = transform

        self.grabber = grabber

    def __len__(self):
        return self.data_hashes.shape[0]

    def __getitem__(self, idx):
        data_hash = self.data_hashes.iloc[idx]

        img = torch.Tensor(self.grabber.image_from_hash(data_hash) / 255)

        class_label = self.class_labels.iloc[idx]
        object_hash = self.object_hashes.iloc[idx]
        obj = self.grabber.object_from_hashes(
            data_hash=data_hash, object_hash=object_hash
        )
        bbox = obj.bounding_box
        h, w, c = img.shape
        y0 = int(bbox.y * h)
        x0 = int(bbox.x * w)
        y1 = int((bbox.y + bbox.h) * h)
        x1 = int((bbox.x + bbox.w) * w)

        cropped_img = img[y0:y1, x0:x1, :].permute(2, 0, 1)
        cropped_img = F.resize(cropped_img, [256, 256], antialias=True)

        if c > 3:
            logger.debug(
                f"Number of image channels inconsistent: got {c}, cropping to 3"
            )
            cropped_img = cropped_img[:-1]

        out = (cropped_img, class_label)
        if self.transform:
            out = self.transform(out)
        return out


class StatsLog:
    def __init__(self, name):
        self.name = name
        self._clear_current()

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0

        self.train_size = 0
        self.val_size = 0

    def _clear_current(self):
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0

        self.train_size = 0
        self.val_size = 0

    def _print(self, epoch, kwargs):
        string = (
            f"[{self.name}] Epoch: {epoch+1:.0f} ||"
            + f"Train Loss: {self.train_loss:.4f} || "
            + f"Train Acc.: {self.train_acc:.4f} || "
            + f"Val Loss: {self.val_loss:.4f} || "
            + f"Val Acc.: {self.val_acc:.4f} || "
        )
        for key, value in kwargs.items():
            if torch.is_tensor(value) and len(value.shape) == 0:
                value = value.item()
            if isinstance(value, float):
                string += key + f" {value:.4f} || "
            if isinstance(value, str):
                string += key + value + " || "
        logger.info(string)

    def push(self, epoch, **kwargs):
        if self.train_size > 0:
            self.train_loss /= self.train_size
            self.train_acc /= self.train_size

        if self.val_size > 0:
            self.val_loss /= self.val_size
            self.val_acc /= self.val_size

        self.train_loss_list.append(self.train_loss)
        self.train_acc_list.append(self.train_acc)
        self.val_loss_list.append(self.val_loss)
        self.val_acc_list.append(self.val_acc)

        self._print(epoch, kwargs)
        self._clear_current()
