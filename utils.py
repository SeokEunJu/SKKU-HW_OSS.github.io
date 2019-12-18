import os
import torch
import torchvision
from torch.utils import data
import cv2
import numpy as np
from random import randint


class Dataset(data.Dataset):
    def __init__(self, dirs, in_size=64, scale_by=4, augmentation=True):
        self.in_size = in_size
        self.scale_by = scale_by
        self.train_size = self.in_size // self.scale_by
        self.augmentation = augmentation
        self.img_list = []

        for d in dirs:
            getFiles(d, self.img_list)

    def __len__(self):
        # return length of list of images
        return len(self.img_list)

    def __getitem__(self, index):
        # define how to get each item from the list and return it
        # hint: read image file -> augment image -> downsample image -> normalize value -> turn it into tensor
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        if self.augmentation:
            img = augmentation(img)
        lr_img, gt_img = downsample(img, size=(self.input_size, self.input_size))
        lr_img = normalization(lr_img, _from=(0, 255))
        gt_img = normalization(gt_img, _from=(0, 255))
        to_tensor = torchvision.transforms.ToTensor()
        lr_img = to_tensor(lr_img).cuda()
        gt_img = to_tensor(gt_img).cuda()

        name = os.path.basename(img_path)

        return lr_img, gt_img, name




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
    # flipping
    flip_flag = randint(0, 1)
    if flip_flag == 1:
        image = cv2.flip(image, 1)

    # rotation
    rot = randint(0, 359)
    if rot < 90:
        rot = 0
    elif rot < 180:
        rot = 90
    elif rot < 270:
        rot = 180
    else:
        rot = 270

    w = image.shape[1]
    h = image.shape[0]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, rot, scale=1.0)
    if rot == 90 or rot == 270:
        image = cv2.warpAffine(image, M, (h, w))

    elif rot == 180:
        image = cv2.warpAffine(image, M, (w, h))

    return image


def downsample(image, size):
    # make sure the input image is 64 x 64
    if not image.shape == (64, 64):
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)

    h = size[0]
    w = size[1]

    # gaussian blurring
    blur_int = 5  # randint(0, 9) * 2 + 1
    lr_img = cv2.GaussianBlur(image, (blur_int, blur_int), 0, 0, borderType=cv2.BORDER_DEFAULT)

    # downsample to a lower resolution (resizing)
    lr_img = cv2.resize(lr_img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    return lr_img, image


def normalization(image, _from=(0, 255)):
    # if the image range in (0, 255): normalize to range of (0, 1)
    # if the image range in (0, 1): turn it back to range of (0, 255)
    if _from == (0, 255):
            return image / 255

    elif _from == (0, 1):
            return image * 255

    # else: out of range
    raise ValueError('wrong range input: normalization only suppoerts range of (0, 1), and (0, 255)')


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


class example_Dataset(data.Dataset):
    def __init__(self, dirs, in_size=64, scale_by=4, augmentation=True):
        self.in_size = in_size
        self.scale_by = scale_by
        self.train_size = self.in_size // self.scale_by
        self.augmentation = augmentation
        self.img_list = []

        for d in dirs:
            getFiles(d, self.img_list)

    def __len__(self):
        # return length of list of images
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        lr_img = img
        gt_img = img
        name = os.path.basename(img_path)

        return  lr_img, gt_img, name