import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

# 假设你有一个特征矩阵X，其形状为 (n_samples, n_features)
# 其中每行是一个样本的特征向量
ncentroids = 10
CAU_full = torch.load('./anchors_w/SPH_cluster256_centroids_full_{}.pkl'.format(ncentroids))
CAU_full = CAU_full.reshape(ncentroids, 256)
# CAU_full = torch.load('/home/whq/HKUSTGZ/Active_L/MADA-main/features_show/SCH_model_on_SCH_256.pkl')
# CAU_full = CAU_full.reshape(CAU_full.shape[0], 256)

CAU_full2 = torch.load('./features_w/SPH_dataset_objective_vectors_256.pkl')
CAU_full2 = CAU_full2.reshape(CAU_full2.shape[0], 256)


# 创建一个TSNE对象
tsne = TSNE(n_components=2, random_state=42)
CAU_combined = np.concatenate((CAU_full, CAU_full2), axis=0)
# 使用fit_transform方法对特征矩阵进行降维，得到降维后的结果
X_embedded = tsne.fit_transform(CAU_combined)

# 绘制T-SNE图
plt.scatter(X_embedded[len(CAU_full):, 0], X_embedded[len(CAU_full):, 1], c='lightskyblue', label='SPH Feature',s=3, marker='.')
plt.scatter(X_embedded[:len(CAU_full), 0], X_embedded[:len(CAU_full), 1], c='red', label='SPH Ref',s=36, marker='*')

plt.legend()
plt.savefig('/home/whq/HKUSTGZ/Active_L/MADA-main/try/SPH_ac.png', dpi=1200)
