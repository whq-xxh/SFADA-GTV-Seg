import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image

#all sample
class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_set/")

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
            # h5f = h5py.File(self._base_dir + "/PLsph/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1

        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            sample["name"] =case  
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample


#@ whq 随机提取部分case-level sample用于训练  Some randomly selected case-level samples were used for training
class BaseDataSets1(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        
        list_NPC=[]
        with open(self._base_dir+'/train.txt', 'r') as file:
            lines = file.readlines()  
        for line in lines:
            list_NPC.append(line.split('.')[0])
        num_selected = int(len(list_NPC) * 0.1)     #@whq sample_percentage
        random_selected = random.sample(list_NPC, num_selected)

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_set/")
            #@whq 提取部分sample用于训练

            matching_samples = []
            for sample in random_selected:
                matching_samples.extend([ext_sample for ext_sample in self.sample_list if ext_sample.startswith(sample)])
            self.sample_list = matching_samples
            print('len(self.sample_list)=',len(self.sample_list))
            print(random_selected)

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1


        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample


#@ whq 提取SFADA_sample用于训练  Select SFADA_sample for training
class BaseDataSets2(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        ##  txt文件存储了主动选择的样本   The txt file stores the actively selected samples
        list_NPC=[]
        with open('/home/whq/HKUSTGZ/Active_L/SFADA2/selection_list/SMU/Ours/MR.txt', 'r') as file:
            lines = file.readlines()

        if self.split == "train":
            for line in lines:
                list_NPC.append(line.replace("\n",""))
            self.sample_list = list_NPC
            print('len(self.sample_list)=',len(self.sample_list))

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1


        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample

#@ whq 生成伪标签   Generate pseudo-labels
class BaseDataSets3(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform


        # list_NPC_all = os.listdir(self._base_dir + "/training_set/")
        list_NPC=[]
        with open('/home/whq/HKUSTGZ/Active_L/MADA-main/selection_list/256_SPH_al_SCH_ist_0.10.txt', 'r') as file:
            lines = file.readlines()

        if self.split == "test":
            for line in lines:
                list_NPC.append(line.replace("\n",""))
            self.sample_list = list_NPC
            print('len(self.sample_list)=',len(self.sample_list))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "test":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1

        if self.split == "test":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            sample["name"] =case           
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample

#@ whq 生成没有list的伪标签，all-list    Generate pseudo-labels without list, all-list
class BaseDataSets5(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform


        list_NPC_all = os.listdir(self._base_dir + "/training_set/")
        list_NPC=[]
        with open('/home/whq/HKUSTGZ/Active_L/MADA-main/selection_list/semiv1_256_SPH_al_NPC1_ist_0.10.txt', 'r') as file:
            lines = file.readlines()

        if self.split == "test":
            for line in lines:
                list_NPC.append(line.replace("\n",""))
        self.sample_list = [x for x in list_NPC_all if x not in list_NPC]
        print('len(self.sample_list)=',len(self.sample_list))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "test":
            h5f = h5py.File(self._base_dir + "/training_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1

        if self.split == "test":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            sample["name"] =case           
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample


#@ whq 使用pseudo label用于训练      Using pseudo labels for training
class BaseDataSets4(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        
        list_NPC=[]
        with open('/home/whq/HKUSTGZ/Active_L/MADA-main/selection_list/256_SPH_al_SCH_ist_0.10.txt', 'r') as file:
            lines = file.readlines()

        if self.split == "train":
            for line in lines:
                list_NPC.append(line.replace("\n",""))
            #@whq
            list_NPC_all = os.listdir(self._base_dir + "/training_set/")
            # self.sample_list = [x for x in list_NPC_all if x not in list_NPC]    #使用list之外的所有          
            # self.sample_list = list_NPC    @#whq
            self.sample_list =list_NPC_all
            print('len(self.sample_list)=',len(self.sample_list))

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_set/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/PL5_al0.1+pl0.9/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val_set/{}".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        label[label > 0] = 1

        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        if len(label.shape) == 2:
            label = np.rot90(label, k)
            label = np.flip(label, axis=axis).copy()
            return image, label
        elif len(label.shape) == 3:
            new_label = np.zeros_like(label)
            for i in range(new_label.shape[0]):
                new_label[i, ...] = np.rot90(label[i, ...], k)
                new_label[i, ...] = np.flip(label[i, ...], axis=axis).copy()
            return image, new_label
        else:
            Exception("Error")
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    elif len(label.shape) == 3:
        new_label = np.zeros_like(label)
        for i in range(label.shape[0]):
            new_label[i, ...] = ndimage.rotate(
                label[i, ...], angle, order=0, reshape=False)
        return image, new_label
    else:
        Exception("Error")


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_aug = augmentations.cta_apply(
            transforms.ToPILImage()(image), ops_weak)
        # if random.random() > 0.5:
        #     image_aug = augmentations.cta_apply(image_aug, ops_strong)
        label_aug = augmentations.cta_apply(
            transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()
        label_aug[label_aug > 0] = 1

        sample = {
            "image_aug": to_tensor(image_aug),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        # print(image.shape)
        x, y = image.shape

        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {"image": image, "label": label}
        return sample

class RandomGenerator2(object):
    def __init__(self, output_size):
        self.output_size = [256,256]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape

        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # print("image.shape=",image.shape)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        x, y = image.shape

        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(
                label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(
            image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
