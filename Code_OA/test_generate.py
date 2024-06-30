import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from distance_metrics_fast import hd95_fast, asd_fast,nsd
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation



parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/whq/HKUSTGZ/Seg_c/data/APH", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="SPH_to_APH_unet", help="experiment_name")
parser.add_argument('--model', type=str,
                    default='SPH', help='data_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--checkpoint', type=str,  default="best",
                    help='last or best')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # print("pred.shape=",pred.shape)
    dice = metric.binary.dc(pred, gt)
    hd95 = hd95_fast(pred, gt, (3, 0.5, 0.5))
    asd = asd_fast(pred, gt, (3, 0.5, 0.5))
    # sur_dic = compute_surface_dice(pred, gt, (3, 0.5, 0.5), tolerance_mm=1.0)
    nsds =nsd(pred, gt, (3, 0.5, 0.5))
    return dice, hd95, asd,nsds


def test_single_volume_fast(case, net, classes, FLAGS, test_save_path, patch_size=[256, 256], batch_size=24):
    h5f = h5py.File(FLAGS.root_path + "/test_set/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f["label"][:]
    label[label > 0] = 1
    spacing = h5f["voxel_spacing"]

    prediction = np.zeros_like(label)

    ind_x = np.array([i for i in range(image.shape[0])])
    for ind in ind_x[::batch_size]:
        if ind + batch_size < image.shape[0]:

            stacked_slices = image[ind:ind + batch_size, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:, ...] = pred
    metric_list = []
    if prediction.max()>0:
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(
                prediction == i, label == i))
        # 保存nii.gz图像   @whq
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing(spacing)
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing(spacing)
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.SetSpacing(spacing)
        sitk.WriteImage(prd_itk, test_save_path +
                        case.replace(".h5", "") + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path +
                        case.replace(".h5", "") + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path +
                        case.replace(".h5", "") + "_gt.nii.gz")
        return np.array(metric_list)
    else:
        print('skip')


def Inference(FLAGS):
    image_list = sorted(os.listdir(FLAGS.root_path + "/test_set"))

    folder_path = "/home/whq/HKUSTGZ/Seg_c/A3_try/model/SPH_unet/"  # 替换为您的文件夹路径

    files = os.listdir(folder_path)
    pth_files = [file for file in files if file.endswith(".pth")]
    sorted_files = sorted(pth_files)
    # with open(folder_path+'output.txt', 'a') as file:

    for file1 in sorted_files:
        if file1 == 'unet_best_model.pth':
            print(folder_path+file1)
            snapshot_path = folder_path+file1
            test_save_path = "/home/whq/HKUSTGZ/Seg_c/A3_try/model/SPH_unet_to_APH_vec256_MR_147_new1_close_adam/all/"
            # if os.path.exists(test_save_path):
            #     shutil.rmtree(test_save_path)
            # os.makedirs(test_save_path)
            net = net_factory(net_type='unet', in_chns=1,
                            class_num=FLAGS.num_classes)
            if FLAGS.checkpoint == "best":
                save_mode_path = snapshot_path
            else:
                save_mode_path = os.path.join(
                    snapshot_path, 'model_iter_60000.pth')
            net.load_state_dict(torch.load(save_mode_path)["state_dict"])
            print("init weight from {}".format(save_mode_path))
            net.eval()

            segmentation_performance = []
            for case in tqdm(image_list):
                metric = test_single_volume_fast(
                    case, net, FLAGS.num_classes, FLAGS, test_save_path)
                segmentation_performance.append(metric)
            segmentation_performance = np.array(segmentation_performance)

            print("model_name = " + snapshot_path + "\n")
            print("dice = mean-sd = " + str(segmentation_performance.mean(axis=0)[0][0]) + "-" + str(segmentation_performance.std(axis=0)[0][0]) + "\n")
            print("hd95 = mean-sd = " + str(segmentation_performance.mean(axis=0)[0][1]) + "-" + str(segmentation_performance.std(axis=0)[0][1]) + "\n")
            print("asd = mean-sd = " + str(segmentation_performance.mean(axis=0)[0][2]) + "-" + str(segmentation_performance.std(axis=0)[0][2]) + "\n")
            print("nsd = mean-sd = " + str(segmentation_performance.mean(axis=0)[0][3]) + "-" + str(segmentation_performance.std(axis=0)[0][3]) + "\n")
            print("\n")

    return segmentation_performance.mean(axis=0), segmentation_performance.std(axis=0)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(FLAGS.root_path)
    print('Model = ',FLAGS.exp)
    print("dice, hd95, asd    (mean-std)")
    print(metric)
