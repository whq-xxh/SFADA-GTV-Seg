import os
import random
import glob
import numpy as np


hopital = "SCH"
all_cases = os.listdir(
    "/home/luoxiangde/Projects/Seg_NPC/data/{}/imagesTr".format(hopital))
random.shuffle(all_cases)
training_set = random.sample(all_cases, int(len(all_cases) * 0.7))
val_testing_set = [i for i in all_cases if i not in training_set]
val_set = random.sample(val_testing_set, int(len(all_cases) * 0.1))
test_set = [i for i in val_testing_set if i not in val_set]

with open("/home/luoxiangde/Projects/Seg_NPC/data/{}/train.txt".format(hopital), "w") as f1:
    for i in training_set:
        f1.writelines("{}\n".format(i))
f1.close()

with open("/home/luoxiangde/Projects/Seg_NPC/data/{}/val.txt".format(hopital), "w") as f2:
    for i in val_set:
        f2.writelines("{}\n".format(i))
f2.close()

with open("/home/luoxiangde/Projects/Seg_NPC/data/{}/test.txt".format(hopital), "w") as f3:
    for i in test_set:
        f3.writelines("{}\n".format(i))
f3.close()

print(len(training_set), len(val_set), len(test_set))
