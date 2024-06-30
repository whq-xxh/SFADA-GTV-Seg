import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
from skimage import exposure

import numpy as np
def merge_histograms(histograms):
    # 计算所有直方图的长度的平均值
    average_length = int(np.mean([len(hist[0]) for hist in histograms]))
    
    # 创建一个新的直方图长度
    new_values = np.linspace(0, 1, average_length)
    
    # 初始化一个数组来存储调整后的直方图
    merged_histogram = np.zeros((len(histograms), average_length))
    
    # 对每个直方图进行插值或重采样
    for i, hist in enumerate(histograms):
        t_values, t_quantiles = hist
        merged_histogram[i, :] = np.interp(new_values, t_quantiles, t_values)
    
    # 取平均值得到代表性的直方图
    representative_histogram = np.mean(merged_histogram, axis=0)
    
    return new_values, representative_histogram


def match_histogram(image, reference_cdf, reference_bin_centers):
    # Calculate the histogram of the test image
    image_hist, bin_edges = np.histogram(image, bins=len(reference_bin_centers), range=(image.min(), image.max()))
    image_cdf = np.cumsum(image_hist).astype(float)
    image_cdf /= image_cdf[-1]

    # Match the histogram of the test image to the reference
    matched_image = np.interp(image.flat, bin_edges[:-1], reference_cdf)
    
    return matched_image.reshape(image.shape)
def hist_match(source, value,qua):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    oldshape = source.shape
    source = source.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # template = template.ravel()
    # t_values, t_counts = np.unique(template, return_counts=True)
    # t_quantiles = np.cumsum(t_counts).astype(np.float64)
    # t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, qua, value)
    return interp_t_values[bin_idx].reshape(oldshape)

def hist_cal(template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    template = template.ravel()
    t_values, t_counts = np.unique(template, return_counts=True)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    return t_values,t_quantiles

hopital1 = "SMU"
with open("/home/whq/HKUSTGZ/Seg_c/data/{}/val.txt".format(hopital1), "r") as f:
    test_set = [i.replace("\n", "") for i in f.readlines()]
f.close()

hopital2 = "WCH"
with open("/home/whq/HKUSTGZ/Seg_c/data/{}/test.txt".format(hopital2), "r") as f:
    test_set2 = [i.replace("\n", "") for i in f.readlines()]
f.close()

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

# image_array1=sitk.GetArrayFromImage(sitk.ReadImage("/home/whq/HKUSTGZ/Seg_c/data/SPH/imagesTr/MR1201605040064_20160506.nii.gz"))
test_num = 0
cdf_list = []
histogram_data = []
bin_list = []
for case in test_set:
    image = "/home/whq/HKUSTGZ/Seg_c/data/{}/imagesTr/{}".format(hopital1, case)
    label = "/home/whq/HKUSTGZ/Seg_c/data/{}/labelsTr/{}".format(hopital1, case)
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)


    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)
    print("image_array.shape=",image_array.shape)
    tv,tq=hist_cal(image_array)
    histogram_data.append((tv, tq))

average_t_quantiles,average_t_values=merge_histograms(histogram_data)
print("t_values.shape",average_t_values.shape)
print(average_t_values)
print("t_quantiles.shape",average_t_quantiles.shape)
print(average_t_quantiles)


# def find_real_filename(directory, case_prefix):
#     for filename in os.listdir(directory):
#         if filename.startswith(case_prefix):
#             return filename
#     return None  # 如果没有找到匹配的文件


test_num = 0
for case in test_set2:
    # directory = '/home/whq/HKUSTGZ/Seg_c/data/APH/imagesTr'  # 替换为您的目录路径
    # real_filename = find_real_filename(directory, case[:7])   
    # print(case[:7]) 
    image = "/home/whq/HKUSTGZ/Seg_c/data/{}/imagesTr/{}".format(hopital2, case)
    label = "/home/whq/HKUSTGZ/Seg_c/data/{}/labelsTr/{}".format(hopital2, case)
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)
    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)
    # matched_image = hist_match(image_array,image_array1)
    matched_image = hist_match(image_array,average_t_values,average_t_quantiles)
    image_array_recorrected = MedicalImageDeal(matched_image, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()
    # matched_image = match_histogram(image_array_recorrected_norm, cdf_list[6],bin_list[6])
    
    f = h5py.File('/home/whq/HKUSTGZ/Seg_c/data/{}/test_set_SMUv/{}.h5'.format(hopital2, case.replace(".nii.gz", "")), 'w')
    f.create_dataset(
        'image', data=image_array_recorrected_norm, compression="gzip")
    f.create_dataset('label', data=label_array, compression="gzip")
    f.create_dataset('voxel_spacing', data=image_itk.GetSpacing(), compression="gzip")
    f.close()
    test_num += 1
print("Converted all NPC volumes to h5 volumes")
print("Total {} volumes".format(test_num))
