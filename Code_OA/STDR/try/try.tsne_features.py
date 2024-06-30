import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

# 假设你有一个特征矩阵X，其形状为 (n_samples, n_features)
# 其中每行是一个样本的特征向量
ncentroids = 10
CAU_full = torch.load('./anchors_w/SPH_cluster256_centroids_full_{}.pkl'.format(ncentroids))
CAU_full = CAU_full.reshape(ncentroids, 256)

CAU_full2 = torch.load('./features_w/SPH_dataset_objective_vectors_256.pkl')
CAU_full2 = CAU_full2.reshape(CAU_full2.shape[0], 256)

CAU_full3 = torch.load('./features_show/SPH_on_APH_256.pkl')
CAU_full3 = CAU_full3.reshape(CAU_full3.shape[0], 256)



# 创建一个TSNE对象
tsne = TSNE(n_components=2, random_state=42)
CAU_combined = np.concatenate((CAU_full, CAU_full2, CAU_full3), axis=0)
# 使用fit_transform方法对特征矩阵进行降维，得到降维后的结果
X_embedded = tsne.fit_transform(CAU_combined)

# 绘制T-SNE图
plt.scatter(X_embedded[len(CAU_full):len(CAU_full)+len(CAU_full2), 0], X_embedded[len(CAU_full):len(CAU_full)+len(CAU_full2), 1], c='cyan', label='SPH',s=6, marker='.')
plt.scatter(X_embedded[len(CAU_full)+len(CAU_full2):, 0], X_embedded[len(CAU_full)+len(CAU_full2):, 1], c='purple', label='SPH_on_APH',s=6, marker='.')
plt.scatter(X_embedded[:len(CAU_full), 0], X_embedded[:len(CAU_full), 1], c='red', label='Achors',s=13, marker='*')

plt.legend()
plt.savefig('/home/whq/HKUSTGZ/Active_L/MADA-main/try/image2/SPH_and_SPH_on_APH.png', dpi=1200)
