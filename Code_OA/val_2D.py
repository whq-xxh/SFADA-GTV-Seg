import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
# from distance_metrics_fast import hd95_fast


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = hd95_fast(pred, gt, (3.0, 0.5, 0.5))
        return dice
    else:
        return 0


def test_single_volume_fast(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    # print("val-image.shape=",image.shape)
    # print("label.shape=",label.shape)

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
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_fast_multi_class(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    label = label.sum(dim=1)
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

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
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_fast_multi_ouputs(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

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
                output_list = net(input)
                for rater_id in range(len(output_list)):
                    out = torch.argmax(torch.softmax(
                        output_list[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                output_list = net(input)
                for rater_id in range(len(output_list)):
                    out = torch.argmax(torch.softmax(
                        output_list[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:, ...] = pred

    metric_array = np.zeros((label.shape[0], classes-1))

    for i in range(1, classes):
        for rater_id in range(label.shape[0]):
            metric_array[rater_id, i-1] = calculate_metric_percase(
                prediction[rater_id, ...] == i, label[rater_id, ...] == i)

    return metric_array


def test_single_volume_fast_padl(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

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
                global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = net(
                    input, training=False)
                for rater_id in range(rater_samples.shape[0]):
                    out = torch.argmax(torch.softmax(
                        rater_samples[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = net(
                    input, training=False)
                for rater_id in range(rater_samples.shape[0]):
                    out = torch.argmax(torch.softmax(
                        rater_samples[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:, ...] = pred

    metric_array = np.zeros((label.shape[0], classes-1))

    for i in range(1, classes):
        for rater_id in range(label.shape[0]):
            metric_array[rater_id, i-1] = calculate_metric_percase(
                prediction[rater_id, ...] == i, label[rater_id, ...] == i)

    return metric_array


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
