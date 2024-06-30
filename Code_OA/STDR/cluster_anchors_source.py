import os
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from utils.clustering import run_kmeans

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    seed = 36
    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load and transition to 24966*(19*256)
    CAU = torch.load('features_w/SPH_vectors_256.pkl')
    print("CAU.shape = ", CAU.shape)
    x = np.reshape(CAU, (CAU.shape[0], CAU.shape[1] * CAU.shape[2])).astype('float32')
    print("CAU.shape = ", x.shape)

    # kmeans
    ncentroids = 30
    cluster_centroids, cluster_index, cluster_loss = run_kmeans(x, ncentroids, verbose=True)

    '''
    # origin cluster
    ncentroids = 10
    niter = 20
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True, gpu=True)
    kmeans.train(x)
    # get the result
    cluster_result = kmeans.centroids
    cluster_loss = kmeans.obj
    '''
    print("cluster_centroids = ",cluster_centroids)
    print("len(cluster_centroids) = ", len(cluster_centroids))
    # print("cluster_index = ",cluster_index)
    print("len(cluster_index) = ",len(cluster_index))
    print("cluster_loss = ",cluster_loss)
    torch.save(cluster_centroids, 'Reference_points_w/SPH_cluster256_centroids_full_%d.pkl' % ncentroids)
    '''
    # import cluster
    nmb = 10
    deepcluster = clustering.Kmeans(nmb)
    clustering_loss = deepcluster.cluster(CAU, verbose=True)
    '''


if __name__ == '__main__':
    main()
