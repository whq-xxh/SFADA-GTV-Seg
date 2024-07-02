# Dual-Reference Source-Free Active Domain Adaptation for Nasopharyngeal Carcinoma Tumor Segmentation across Multiple Hospitals 🏥

We are excited to announce that our paper was accepted for publication at **IEEE TMI 2024**! 🥳🥳🥳

This repository contains the official implementation of our paper. 
You can access the paper [here](https://ieeexplore.ieee.org/abstract/document/10553522).

# Introduction 📑

This project introduces a new setting in medical image segmentation, termed **Source-Free Active Domain Adaptation (SFADA)**. SFADA aims to facilitate cross-center medical image segmentation while protecting data privacy and reducing the workload on medical professionals. By requiring only minimal labeling effort, SFADA achieves effective model transfer and results comparable to those of fully supervised approaches.

Fig. 1. Visual comparison of traditional training and our Source-Free Active Domain Adaptation (SFADA) training.
<img width="800" alt="compa" src="https://github.com/whq-xxh/Active-GTV-Seg/assets/119860058/faea09fc-2437-434d-a332-356529a101ea">

# How to Run the Code 🛠
## Environment Installation
`conda create --name SFADA --file Code_OA/requirements.txt`
### Convert nii.gz Files to h5 Format to facilitate follow-up processing and training🔄
`python dataloaders/data_processing.py`

### 1. Training source models in a single center
`python train_single_center.py`

### 2. Run inference and save latent space representations 
To perform inference and save the latent space representations of all samples, use the following command:

`python STDR/save_source.py`

### 3. Cluster the Reference Points R^s 🌟

To cluster out the reference points R^s from the latent space representations, run the following command:

`python STDR/cluster_anchors_source.py`

### 4. Select Active Samples Using STDR Strategy 🎯

This step is to select the actively labeled samples based on our STDR strategy. The source model is used to infer the latent space representations of all the samples in the target center, and the samples are selected based on Reference Points R^s 🌟 and our STDR strategy.

`python STDR/select_active_samples_w_256.py`

### 5. Finetune the source Model with actively labeled samples 🔧

To fine-tune the source model using the actively labeled samples selected through the STDR strategy, run the following command:

`python train_single_center_finetune.py`

The results of this model correspond to the results of the STDR in the paper.

### 6. Others

Test the model with `python test_single_center.py`. `python test_generate.py` can be used to infer the pseudo-labels of the samples, combining the pseudo-labels with the actively labeled samples in a common model-finetuning, to get the final result of "Ours" in our paper.

*Feel free to contact my email (hongqiuwang16@gmail.com) with any questions on reproduction.*

# Dataset 📊
In the Discussion section of our paper, we mentioned our efforts to construct a relevant dataset. We are pleased to offer access to this dataset, which includes anonymized data from three centers: **Center A (50 cases)** 🏥, **Center B (50 cases)** 🏨, and **Center C (60 cases)** 🏬. We invite researchers working on **multi-center segmentation** and **GTV segmentation** to make use of this valuable resource. 

Please contact Hongqiu (hongqiuwang16@gmail.com) for the dataset. One step is needed to download the dataset: **1) Use your google email to apply for the download permission ([Goole Driven](https://drive.google.com/drive/folders/1Oc6l11BRmkLfVwHW_WnnYzG0Eu2AFsB-)). We will get back to you within three days, so please don't send them multiple times. We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:

    Name/Homepage/Google Scholar: (Tell us who you are.)
    Primary Affiliation: (The name of your institution or university, etc.)
    Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
    Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
    How to use: (Only for academic research, not for commercial use or second-development.)

# Citation 📖

If you find our work useful or relevant to your research, please consider citing:
```
@article{wang2024dual,
  title={Dual-Reference Source-Free Active Domain Adaptation for Nasopharyngeal Carcinoma Tumor Segmentation across Multiple Hospitals},
  author={Wang, Hongqiu and Chen, Jian and Zhang, Shichen and He, Yuan and Xu, Jinfeng and Wu, Mengwan and He, Jinlan and Liao, Wenjun and Luo, Xiangde},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```

# Comparison with Other Methods 📈

*Details on other comparison approaches will be added soon.*
