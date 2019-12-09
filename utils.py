import os
import torch
import torchvision
from torch.utils import data
import cv2
import numpy as np
from random import randint


class Dataset(data.Dataset):
    def __init__(self, dirs, in_size=64, scale_by=4):
        self.in_size = in_size
        self.scale_by = scale_by
        self.train_size = self.in_size // self.scale_by
        self.img_list = []

        for d in dirs:
            getFiles(d, self.img_list)

    def __len__(self):
        # return length of list of images
        return len(self.img_list)

    def __getitem__(self, index):
        # define how to get each item from the list and return it
        # hint: read image file -> augment image -> downsample image -> normalize value -> turn it into tensor

        return  # lr_img, gt_img, name




# get all image files within the directory 'dir'
def getFiles(dir, dataList):
    if os.path.isdir(dir):
        temp_dataList = os.listdir(dir)
        for directory in temp_dataList:
            directory = os.path.join(dir, directory)
            getFiles(directory, dataList)
    elif os.path.isfile(dir):
        if dir.endswith('.png') or dir.endswith('.jpeg') or dir.endswith('.jpg'):
            dataList.append(dir)


def augmentation(image):
    # random flipping

    # random rotation

    return image


def downsample(image, size):
    # size input in a tuple form (h , w)

    # random blur

    # downsample to make a LR image, and return original image together

    return lr_img, image


def normalization(image, _from=(0, 255)):
    # if the image range in (0, 255): normalize to range of (0, 1)
    # if the image range in (0, 1): turn it back to range of (0, 255)

    # else: out of range
    raise ValueError('wrong range input: normalization only suppoerts range of (0, 1) and (0, 255)')


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

