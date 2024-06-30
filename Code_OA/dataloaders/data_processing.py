import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

hopital = "I2CVB"

with open("/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/train.txt".format(hopital), "r") as f:
    training_set = [i.replace("\n", "") for i in f.readlines()]
f.close()

with open("/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/val.txt".format(hopital), "r") as f:
    val_set = [i.replace("\n", "") for i in f.readlines()]
f.close()


with open("/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/test.txt".format(hopital), "r") as f:
    test_set = [i.replace("\n", "") for i in f.readlines()]
f.close()


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

slice_num = 0
for case in training_set:
    image = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/imagesTr/{}".format(hopital, case)
    label = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/labelsTr/{}".format(hopital, case.replace('.nii','_segmentation.nii'))
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)


    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)

    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()

    for slice_ind in range(image_array.shape[0]):
        f = h5py.File('/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/training_set/{}_slice_{}.h5'.format(hopital, case.replace(".nii.gz", ""), slice_ind), 'w')
        f.create_dataset(
            'image', data=image_array_recorrected_norm[slice_ind], compression="gzip")
        f.create_dataset('label', data=label_array[slice_ind], compression="gzip")
        f.close()
        slice_num += 1
print("Converted all NPC volumes to 2D slices")
print("Total {} slices".format(slice_num))

val_num = 0
for case in val_set:
    image = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/imagesTr/{}".format(hopital, case)
    label = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/labelsTr/{}".format(hopital, case.replace('.nii','_segmentation.nii'))
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)


    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)

    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()

    
    f = h5py.File('/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/val_set/{}.h5'.format(hopital, case.replace(".nii.gz", "")), 'w')
    f.create_dataset(
        'image', data=image_array_recorrected_norm, compression="gzip")
    f.create_dataset('label', data=label_array, compression="gzip")
    f.create_dataset('voxel_spacing', data=image_itk.GetSpacing(), compression="gzip")
    f.close()
    val_num += 1
print("Converted all NPC volumes to h5 volumes")
print("Total {} volumes".format(val_num))


test_num = 0
for case in test_set:
    image = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/imagesTr/{}".format(hopital, case)
    label = "/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/labelsTr/{}".format(hopital, case.replace('.nii','_segmentation.nii'))
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)


    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)

    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()

    
    f = h5py.File('/home/whq/HKUSTGZ/Seg_c/A1.1/data2/{}/test_set/{}.h5'.format(hopital, case.replace(".nii.gz", "")), 'w')
    f.create_dataset(
        'image', data=image_array_recorrected_norm, compression="gzip")
    f.create_dataset('label', data=label_array, compression="gzip")
    f.create_dataset('voxel_spacing', data=image_itk.GetSpacing(), compression="gzip")
    f.close()
    test_num += 1
print("Converted all NPC volumes to h5 volumes")
print("Total {} volumes".format(test_num))

