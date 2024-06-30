import argparse
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
from tqdm import tqdm
from utils.utils import get_logger
from tensorboardX import SummaryWriter
import heapq
import torch.nn as nn


#@ whq
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from dataloaders_w.dataset import (
    BaseDataSets,
    CTATransform,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def Savefeat(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    db_train = BaseDataSets(
    base_dir="/home/whq/HKUSTGZ/Seg_c/data/APH",
    split="train",
    transform=RandomGenerator([256, 256]))
    
    train_loader = DataLoader(db_train, batch_size=1, shuffle=False,
                            num_workers=16, pin_memory=True)
    # create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')

    def create_model():
        model = net_factory(net_type="unet", in_chns=1,
                            class_num=2)
        return model
    model = create_model()
    model.load_state_dict(torch.load('/home/whq/HKUSTGZ/Seg_c/model/SPH_NPC_unet/unet_best_model.pth')["state_dict"])

    class_features = Class_Features(numbers=2)
    ncentroids = 40
    CAU_full = torch.load('./anchors_w/SPH_cluster256_centroids_full_{}.pkl'.format(ncentroids))
    CAU_full = CAU_full.reshape(ncentroids, 1, 256)
    class_features.centroids = CAU_full

    i_iter = 0
    cac_list = []
    case_list = []
    print(len(train_loader))

    with torch.no_grad():
        # for epoch_num in iterator:
        for idx, sampled_batch in enumerate(train_loader):
            image_batch, label_batch, case  = (
                sampled_batch["image"],
                sampled_batch["label"], 
                sampled_batch["name"],
            )
            image_batch, labels = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            model.eval()
            feat_cls, output = model(image_batch)      
            batch, w, h = labels.size()
            newlabels = labels.reshape([batch, 1, w, h]).float()
            newlabels = F.interpolate(newlabels, size=feat_cls.size()[2:], mode='nearest')
            vectors, ids = class_features.calculate_mean_vector(feat_cls, output, newlabels, model)
            # print("vectors[0].shape = ",vectors.shape)
            # print("ids = ",ids)
            single_image_objective_vectors = np.zeros([1, 256])
            case_list.append(case)

            for t in range(len(ids)):
                # print("t=",t)
                single_image_objective_vectors[ids[t]] = vectors[t].detach().cpu().numpy().squeeze()
                # model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
            MSE = class_features.calculate_min_mse(single_image_objective_vectors)
            cac_list.append(MSE)

    lenth = len(train_loader)
    per = 0.1
    selected_lenth = int(per * lenth)
    selected_index_list = list(map(cac_list.index, heapq.nlargest(selected_lenth, cac_list)))
    selected_index_list2 = list(map(cac_list.index, heapq.nsmallest(selected_lenth, cac_list)))           #@whq STDR strategy
    selected_img_list = []
    for index in selected_index_list:
        selected_img_list.append(case_list[index])
    for index in selected_index_list2:
        selected_img_list.append(case_list[index])
    print("len(selected_img_list)=",len(selected_img_list))
    
    file = open(os.path.join('./selection_list', 'semiv1_256_SPH_al_APH_ist_%.2f_.txt' % per), 'w')     #@whq Selection list
    for i in range(len(selected_img_list)):
        img = str(selected_img_list[i])
        img = img.strip("[]'")
        file.write(img + '\n')
    file.close()


class Class_Features:
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        # self.class_features = np.zeros((19, 256))
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.all_vectors = []
        self.pred_ids = []
        self.ids = []
        self.pred_num = np.zeros(numbers + 1)
        return

    def calculate_mean_vector(self, feat_cls, outputs, labels_val, model):
        # print("outputs.shape=",outputs.shape)
        # print(outputs.max())
        
        outputs_softmax = F.softmax(outputs, dim=1)
        tensor1, tensor2 = torch.split(outputs_softmax, 1, dim=1)
        outputs_argmax=tensor1.float()
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        # print("scale_factor=",scale_factor)
        vectors = []
        ids = []
        # print("feat_cls.size()[0]=",feat_cls.size()[0])
        for n in range(feat_cls.size()[0]):
            for t in range(1):
                # print("t=",t)
                if scale_factor[n][t].item() == 0:
                    print('skip')
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    print('skip2')
                    continue
                s = feat_cls[n] * outputs_pred[n][t]

                s = torch.mean(s, dim=0).unsqueeze(0)
                max_pool = nn.MaxPool2d(kernel_size=16)
                # 进行最大池化操作
                output = max_pool(s)
                # 展平
                s = output.view(output.size(0), -1)/ scale_factor[n][t]

                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = np.mean((single_image_objective_vectors - centroid) ** 2)
            loss.append(new_loss)
        min_loss = min(loss)
        min_index = loss.index(min_loss)
        # print(min_loss)
        # print(min_index)
        return min_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/SPH_to_SCH_source.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # run_id = random.randint(1, 100000)
    run_id = 16
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    Savefeat(cfg, writer, logger)