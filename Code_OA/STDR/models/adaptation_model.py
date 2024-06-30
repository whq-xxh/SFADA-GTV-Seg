import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import get_loss_function
from models.deeplab import DeepLab
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from schedulers import get_scheduler
from .utils import normalisation_pooling


class CustomModel:
    def __init__(self, cfg, writer, logger):
        # super(CustomModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.class_numbers = 19
        self.logger = logger
        cfg_model = cfg['model']
        self.cfg_model = cfg_model
        self.best_iou = -100
        self.iter = 0
        self.nets = []
        self.split_gpu = 0
        self.default_gpu = cfg['model']['default_gpu']
        self.PredNet_Dir = None
        self.valid_classes = cfg['training']['valid_classes']
        self.G_train = True
        self.cls_feature_weight = cfg['training']['cls_feature_weight']
        self.centroids = np.zeros((10, 19, 256)).astype('float32')

        bn = cfg_model['bn']
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))
        self.PredNet = DeepLab(
            num_classes=19,
            backbone=cfg_model['basenet']['version'],
            output_stride=16,
            bn=cfg_model['bn'],
            freeze_bn=True,
        ).cuda()
        self.load_PredNet(cfg, writer, logger, dir=None, net=self.PredNet)
        self.PredNet_DP = self.init_device(self.PredNet, gpu_id=self.default_gpu, whether_DP=True)
        self.PredNet.eval()
        self.PredNet_num = 0

        self.BaseNet = DeepLab(
            num_classes=19,
            backbone=cfg_model['basenet']['version'],
            output_stride=16,
            bn=cfg_model['bn'],
            freeze_bn=False,
        )

        logger.info('the backbone is {}'.format(cfg_model['basenet']['version']))

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets.extend([self.BaseNet])
        self.nets_DP = [self.BaseNet_DP]

        self.optimizers = []
        self.schedulers = []
        # optimizer_cls = get_optimizer(cfg)
        optimizer_cls = torch.optim.SGD
        optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                            if k != 'name'}
        # optimizer_cls_D = torch.optim.SGD
        # optimizer_params_D = {k:v for k, v in cfg['training']['optimizer_D'].items() 
        #                     if k != 'name'}
        self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
        self.schedulers.extend([self.BaseSchedule])
        self.setup(cfg, writer, logger)

        self.adv_source_label = 0
        self.adv_target_label = 1
        self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        self.loss_fn = get_loss_function(cfg)
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.smoothloss = nn.SmoothL1Loss()
        self.triplet_loss = nn.TripletMarginLoss()

    def create_PredNet(self, ):
        ss = DeepLab(
            num_classes=19,
            backbone=self.cfg_model['basenet']['version'],
            output_stride=16,
            bn=self.cfg_model['bn'],
            freeze_bn=True,
        ).cuda()
        ss.eval()
        return ss

    def setup(self, cfg, writer, logger):
        '''
        set optimizer and load pretrained model
        '''
        for net in self.nets:
            # name = net.__class__.__name__
            self.init_weights(cfg['model']['init'], logger, net)
            print("Initializition completed")
            if hasattr(net, '_load_pretrained_model') and cfg['model']['pretrained']:
                print("loading pretrained model for {}".format(net.__class__.__name__))
                net._load_pretrained_model()
        '''load pretrained model
        '''
        if cfg['training']['resume_flag']:
            self.load_nets(cfg, writer, logger)
        pass

    def forward(self, input):
        feat, feat_low, feat_cls, output = self.BaseNet_DP(input)
        return feat, feat_low, feat_cls, output

    def forward_Up(self, input):
        feat, feat_low, feat_cls, output = self.forward(input)
        output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        return feat, feat_low, feat_cls, output

    def PredNet_Forward(self, input):
        with torch.no_grad():
            _, _, feat_cls, output_result = self.PredNet_DP(input)
        return _, _, feat_cls, output_result

    def calculate_mean_vector(self, feat_cls, outputs, labels, ):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        labels_expanded = self.process_label(labels)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
                s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def step_active_stage1(self, source_x, source_label, target_x, target_label, active_x, active_label):
        _, _, active_feat_cls, active_output = self.forward(input=active_x)
        active_outputUp = F.interpolate(active_output, size=active_x.size()[2:], mode='bilinear', align_corners=True)

        loss_active = self.loss_fn(input=active_outputUp, target=active_label)
        if loss_active.item() != 0:
            loss_active.backward()

        _, _, source_feat_cls, source_output = self.forward(input=source_x)
        source_outputUp = F.interpolate(source_output, size=source_x.size()[2:], mode='bilinear', align_corners=True)

        loss_GTA = self.loss_fn(input=source_outputUp, target=source_label)
        self.PredNet.eval()

        loss = torch.Tensor([0]).cuda()
        loss = loss + loss_GTA
        if loss.item() != 0:
            loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()
        return loss, loss_active

    def step_active_stage2(self, source_x, source_label, target_x, target_label, active_x, active_label):
        _, _, active_feat_cls, active_output = self.forward(input=active_x)
        active_outputUp = F.interpolate(active_output, size=active_x.size()[2:], mode='bilinear', align_corners=True)

        loss_active = self.loss_fn(input=active_outputUp, target=active_label)
        if loss_active.item() != 0:
            loss_active.backward()

        _, _, source_feat_cls, source_output = self.forward(input=source_x)
        source_outputUp = F.interpolate(source_output, size=source_x.size()[2:], mode='bilinear', align_corners=True)

        loss_GTA = self.loss_fn(input=source_outputUp, target=source_label)
        self.PredNet.eval()

        loss_L2_source_cls = torch.Tensor([0]).cuda(self.split_gpu)
        loss_L2_target_cls = torch.Tensor([0]).cuda(self.split_gpu)
        _, _, target_feat_cls, target_output = self.forward(target_x)

        if self.cfg['training']['loss_L2_cls']:  # distance loss
            _batch, _w, _h = source_label.shape
            source_label_downsampled = source_label.reshape([_batch, 1, _w, _h]).float()
            source_label_downsampled = F.interpolate(source_label_downsampled.float(), size=source_feat_cls.size()[2:],
                                                     mode='nearest')  # or F.softmax(input=source_output, dim=1)
            # source map
            source_vectors, source_ids = self.calculate_mean_vector(source_feat_cls, source_output,
                                                                    source_label_downsampled)
            source_objective_vectors = torch.zeros([19, 256]).cuda()
            for t in range(len(source_ids)):
                source_objective_vectors[source_ids[t]] = source_vectors[t].squeeze()
            # target map
            target_outputs_softmax = F.softmax(target_output, dim=1)
            target_outputs_argmax = target_outputs_softmax.argmax(dim=1, keepdim=True)
            target_vectors, target_ids = self.calculate_mean_vector(target_feat_cls, target_output,
                                                                    target_outputs_argmax.float())
            target_objective_vectors = torch.zeros([19, 256]).cuda()
            for t in range(len(target_ids)):
                target_objective_vectors[target_ids[t]] = target_vectors[t].squeeze()
            # calculate min distance
            loss_L2_source_cls, _ = self.calculate_min_mse(source_objective_vectors)
            loss_L2_target_cls, min_index = self.calculate_min_mse(target_objective_vectors)

            # update features
            self.centroids[min_index] = self.centroids[
                                            min_index] * 0.999 + 0.001 * target_objective_vectors.detach().cpu().numpy()

        loss_L2_cls = self.cls_feature_weight * loss_L2_target_cls

        loss = torch.Tensor([0]).cuda()
        batch, _, w, h = target_outputs_argmax.shape
        # cluster_arg[cluster_arg != threshold_arg] = 250
        loss_CTS = self.loss_fn(input=target_output, target=target_outputs_argmax.reshape([batch, w, h]))

        if self.G_train and self.cfg['training']['loss_pseudo_label']:
            loss = loss + loss_CTS
        if self.G_train and self.cfg['training']['loss_source_seg']:
            loss = loss + loss_GTA
        if self.cfg['training']['loss_L2_cls']:
            loss = loss + torch.sum(loss_L2_cls)

        if loss.item() != 0:
            loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()
        return loss, loss_L2_cls.item(), loss_CTS.item(), loss_active.item()

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = torch.mean((single_image_objective_vectors - torch.Tensor(centroid).cuda()) ** 2)
            loss.append(new_loss)

        min_loss = min(loss)
        min_index = loss.index(min_loss)

        sum_loss = sum(loss)
        weights = []
        weighted_loss = []
        for item in loss:
            weight = 1 / item
            weighted_loss.append(weight * item)
            weights.append(weight)
        return sum(weighted_loss) / sum(weights), min_index

    def scheduler_step(self):
        # for net in self.nets:
        #     self.schedulers[net.__class__.__name__].step()
        for scheduler in self.schedulers:
            scheduler.step()

    def optimizer_zerograd(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizer_step(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].step()
        for opt in self.optimizers:
            opt.step()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net

    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger != None:
                logger.info("Successfully set the model eval mode")
        else:
            net.eval()
            if logger != None:
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net == None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
            # if logger!=None:    
            #     logger.info("Successfully set the model train mode") 
        else:
            net.train()
            # if logger!= None:
            #     logger.info(print("Successfully set {} train mode".format(net.__class__.__name__)))
        return

    def init_weights(self, cfg, logger, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, SynchronizedBatchNorm2d) or classname.find('BatchNorm2d') != -1 \
                    or isinstance(m, nn.GroupNorm):
                # or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

        print('initialize {} with {}'.format(init_type, net.__class__.__name__))
        logger.info('initialize {} with {}'.format(init_type, net.__class__.__name__))
        net.apply(init_func)  # apply the initialization function <init_func>
        pass

    def adaptive_load_nets(self, net, model_weight):
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in model_weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def load_nets(self, cfg, writer, logger):  # load pretrained weights on the net
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            _k = -1
            for net in self.nets:
                name = net.__class__.__name__
                _k += 1
                if checkpoint.get(name) == None:
                    continue
                if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                    continue
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if cfg['training']['optimizer_resume']:
                    self.adaptive_load_nets(self.optimizers[_k], checkpoint[name]["optimizer_state"])
                    self.adaptive_load_nets(self.schedulers[_k], checkpoint[name]["scheduler_state"])
            self.iter = checkpoint["iter"]
            self.best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["iter"]
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(cfg['training']['resume']))

    def load_PredNet(self, cfg, writer, logger, dir=None, net=None):  # load pretrained weights on the net
        dir = dir or cfg['training']['Pred_resume']
        best_iou = 0
        if os.path.isfile(dir):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(dir)
            )
            checkpoint = torch.load(dir)
            name = net.__class__.__name__
            if checkpoint.get(name) == None:
                return
            if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                return
            self.adaptive_load_nets(net, checkpoint[name]["model_state"])
            iter = checkpoint["iter"]
            best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {}) (best iou {}) for PredNet".format(
                    dir, checkpoint["iter"], best_iou
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(dir))
        if hasattr(net, 'best_iou'):
            net.best_iou = best_iou
        return best_iou
