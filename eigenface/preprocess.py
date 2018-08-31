import os
import numpy as np

def calculate_dice(imgs, seed, spc, margin, mask):
    return 1

def cluster(ori, now):
    return True

all_dice = []
for idx in range(500):
    dice_list = []
    temp_ct = mask_list[idx]
    img_path = temp_ct[0,0]
    img_path = img_path.replace('/data', '/data2')
    imgs, spc, uid = ReadDICOMFolder(img_path)
    for i in range(len(temp_ct)):
        nodule_temp = temp_ct[i]
        mask_path = nodule_temp[1]
        try:
            mask = np.load('/data3/cube'+mask_path)
        except Exception, err:
            continue
        margin = max(int(nodule_temp[10]- nodule_temp[7]), int(nodule_temp[9])-int(nodule_temp[6]))
        seed = [int(nodule_temp[2]), int(nodule_temp[3]), int(nodule_temp[4])]
        dice_temp = calculate_dice(imgs, seed, spc, margin, mask)
        dice_list.append([seed,dice_temp])
        for temp in dice_list:
            seed_temp = temp[0]
            dice_temp = temp[1]
            maximum = dice_temp
            for el in [x for x in dice_list if x is not temp]:
                seed_comp = el[0]
                dice_comp = el[1]
                if cluster(seed_comp, seed_temp):
                    maximum = max(dice_temp, dice_comp)
                    dice_list.remove(el)
        if maximum != 0:
            all_dice.append(maximum)



