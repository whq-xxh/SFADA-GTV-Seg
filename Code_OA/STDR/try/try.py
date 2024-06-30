import h5py
import os

list_NPC=[]
# with open('/home/whq/HKUSTGZ/Active_L/MADA-main/selection_list/256_SMU_close_WCH_ist_0.10.txt', 'r') as file:
#     lines = file.readlines()
list_NPC=os.listdir("/home/whq/HKUSTGZ/Seg_c/data/SCH/" + "/training_set/")

# for line in lines:
#     list_NPC.append(line.replace("\n",""))



count=0
for case in list_NPC:
    h5f = h5py.File("/home/whq/HKUSTGZ/Seg_c/data/SCH" + "/training_set/{}".format(case), "r")
    image = h5f["image"][:]
    label = h5f["label"][:]
    print("label.shape=",label.shape)
    print("image.shape=",image.shape)
#     if label.max()==0:
#         count=count+1

# print('len(list_NPC)=',len(list_NPC))
# print("0 count=",count)