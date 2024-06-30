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
from data import create_dataset
from utils.utils import get_logger
from models.adaptation_model import CustomModel
from metrics import runningScore, averageMeter
from loss import get_loss_function
from tensorboardX import SummaryWriter
import torch.nn as nn

#@ whq
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from dataloaders_w.dataset import (
    BaseDataSets2,
    BaseDataSets,
    CTATransform,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def process_label(label):
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, 3, w, h).cuda()
    id = torch.where(label < 2, label, torch.Tensor([2]).cuda())
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1


def Savefeat(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    db_train = BaseDataSets(
    base_dir="/home/whq/HKUSTGZ/Seg_c/data/WCH",
    split="train",
    transform=RandomGenerator([256, 256]))
    
    train_loader = DataLoader(db_train, batch_size=1, shuffle=False,
                            num_workers=16, pin_memory=True)
    # create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    # datasets = create_dataset(cfg, writer, logger)  # source_train\ target_train\ source_valid\ target_valid + _loader

    # model = CustomModel(cfg, writer, logger)
    def create_model():
    # Network definition
        model = net_factory(net_type="unet", in_chns=1,
                            class_num=2)
        return model
    model = create_model()
    model.load_state_dict(torch.load('/home/whq/HKUSTGZ/Seg_c/model/WCH_NPC_unet/unet_best_model.pth')["state_dict"])

    class_features = Class_Features(numbers=2)

    i_iter = 0
    print(len(train_loader))
    full_dataset_objective_vectors = np.zeros([len(train_loader), 1, 256])

    # iterator = tqdm(range(0, 5684), ncols=120)

    with torch.no_grad():
        # for epoch_num in iterator:
        for batch_idx, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = (
                sampled_batch["image"],
                sampled_batch["label"],
            )
            image_batch, labels = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            model.eval()
            feat_cls, output = model(image_batch)      

            # _, _, feat_cls, output = model.PredNet_Forward(images)       #@ whq 
            # print("output.shape, feat_cls.shape = ",output.shape, feat_cls.shape)
            # print("labels.shape = ",labels.shape)
            batch, w, h = labels.size()
            newlabels = labels.reshape([batch, 1, w, h]).float()
            newlabels = F.interpolate(newlabels, size=feat_cls.size()[2:], mode='nearest')
            vectors, ids = class_features.calculate_mean_vector(feat_cls, output, newlabels, model)
            single_image_objective_vectors = np.zeros([1, 256])

            

            for t in range(len(ids)):
                # print("t=",t)
                single_image_objective_vectors[ids[t]] = vectors[t].detach().cpu().numpy().squeeze()
                # model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
            full_dataset_objective_vectors[i_iter, :] = single_image_objective_vectors[:]
            i_iter += 1
            print("i_iter=",i_iter)
    print('full_dataset_objective_vectors=',full_dataset_objective_vectors.shape)
    torch.save(full_dataset_objective_vectors, 'features_w/WCH_dataset_objective_vectors_256.pkl')
    ##存储所有的样本的隐空间表达     Store the latent space representation of all the samples


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
        outputs_softmax = F.softmax(outputs, dim=1)

        tensor1, tensor2 = torch.split(outputs_softmax, 1, dim=1)
        outputs_argmax=tensor1.float()
        labels_expanded=labels_val
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
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

    run_id = 16
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    Savefeat(cfg, writer, logger)