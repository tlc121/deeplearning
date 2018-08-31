import SimpleITK as sitk
import os
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import disk, binary_dilation
import numpy as np
from scipy.ndimage.interpolation import zoom
from skimage.morphology import convex_hull_image
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

def ReadDICOMFolder(folderName, input_uid=None):
    '''A nearly perfect DCM reader!'''
    reader = sitk.ImageSeriesReader()
    out_uid = ''
    out_image = None
    max_slice_num = 0
    # if the uid is not given, iterate all the available uids
    try:
        uids = [input_uid] if input_uid!=None else reader.GetGDCMSeriesIDs(folderName)
    except TypeError:
        folderName = folderName.encode('utf-8')
        uids = [input_uid] if input_uid!=None else reader.GetGDCMSeriesIDs(folderName)
    for uid in uids:
        try:
            dicomfilenames = reader.GetGDCMSeriesFileNames(folderName,uid)
            reader.SetFileNames(dicomfilenames)
            image = reader.Execute()
            size = image.GetSize()
            if size[0] == size[1] and size[2]!=1: # exclude xray
                slice_num = size[2]
                if slice_num > max_slice_num:
                    out_image = image
                    out_uid = uid
                    max_slice_num = slice_num
        except:
            pass
    if out_image != None:
        imageRaw = sitk.GetArrayFromImage(out_image)
        imageRaw = np.flip(imageRaw, 0)
        spacing = list(out_image.GetSpacing())
        spacing.reverse()
        return imageRaw, np.array(spacing), uid
    else:
        raise Exception('Fail to load the dcm folder.')

def get_chest_boundary(im, selem_area):
    size_row, size_col = im.shape[0], im.shape[1]
    binary = im < -320
    cleared = binary_fill_holes(binary)
    temp_label = label(cleared)
    for region in regionprops(temp_label):
        if region.area < 300:
            for coord in region.coords:
                temp_label[coord[0], coord[1]] = 0
    cleared = temp_label>0
    label_img = label(cleared)
    for region in regionprops(label_img):
        if region.centroid[0] > 0.8 * size_row \
                or region.centroid[0] < 0.2 * size_row \
                or region.centroid[1] > 0.8 * size_col \
                or region.centroid[1] < 0.2 * size_col \
                or (region.centroid[0] < 0.2 * size_row and region.centroid[1] < 0.2 * size_col) \
                or (region.centroid[0] < 0.2 * size_row and region.centroid[1] > 0.8 * size_col) \
                or (region.centroid[0] > 0.8 * size_row and region.centroid[1] < 0.2 * size_col) \
                or (region.centroid[0] > 0.8 * size_row and region.centroid[1] > 0.8 * size_col):

            for coordinates in region.coords:
                label_img[coordinates[0], coordinates[1]] = 0
    label_img = label(label_img)
    region_n = np.max(label_img)
    selem = disk(selem_area)
    filled_dilation = np.zeros(label_img.shape, np.uint8)
    filled_ori = np.zeros(label_img.shape, np.uint8)
    for i in range(1, region_n+1):
        ori = np.zeros(label_img.shape, np.uint8)
        cur_region = np.zeros(cleared.shape, np.uint8)
        temp = (label_img == i)
        temp = binary_dilation(temp, selem)
        ori[label_img == i] = 1
        ori = convex_hull_image(ori)
        cur_region[temp == True] = 1
        filled_dilation[cur_region==1]=1
        filled_ori[ori==1]=1


    filled = filled_dilation - filled_ori
    result = im*filled
    return result, filled_ori

def lung_window(img, wl=300, ww=600):
    img[img < (wl - ww * 0.5)] = wl - ww * 0.5
    img[img > (wl - ww * 0.5)] = wl + ww * 0.5
    return img

def extract(img, area_threshold, hu_threshold, selem):
    empty = np.zeros(img.shape)
    rib, chest = get_chest_boundary(img, selem)
    if np.sum(chest)/(chest.shape[0]*chest.shape[1]) < area_threshold:
        return empty
    else:
        # yy, xx = np.where(chest)
        # min_x, min_y, max_x, max_y = np.min(xx), np.min(yy), np.max(xx), np.max(yy)
        # lungs = lung_window(img)
        # real_ribs = lungs>hu_threshold
        # label_ribs = label(real_ribs)
        # for region in regionprops(label_ribs):
        #     for coord in region.coords:
        #         row, col = coord[0], coord[1]
        #         if row < min_y or row > max_y \
        #             or col < min_x or col > max_x:
        #             for coord in region.coords:
        #                 label_ribs[coord[0], coord[1]] = 0
        #             break
        #         else:
        #             continue
        label_ribs = rib>hu_threshold
        for region in regionprops(label(label_ribs)):
            for coord in region.coords:
                row, col = coord[0], coord[1]
                if chest[row, col] == 1:
                    for coord in region.coords:
                        label_ribs[coord[0], coord[1]] = 0
                    break
                else:
                    continue

        return label_ribs

if __name__ == '__main__':
    img_path = '/Users/xavier0121/Desktop/work/lc/fangguangneng/DICOM/1900813/007-0674-101'
    imgs, spc, uid = ReadDICOMFolder(img_path)
    resize_factor = [spc[0], 1, 1]
    imgs = zoom(imgs, resize_factor, mode = 'nearest', order=2)
    recons = np.zeros(imgs.shape)
    recons2 = np.zeros(imgs.shape)

    centerline = np.zeros(imgs.shape)
    centerline2 = np.zeros(imgs.shape)

    for i in tqdm(range(150, imgs.shape[1])):
        temp = imgs[:,i,:]
        temp2 = imgs[:,:,i]
        res = extract(temp, 0.1, 350, 10)
        res2 = extract(temp2, 0.2, 350, 13)

        recons[:,i,:] = res
        recons2[:,:,i] = res2

    for i in range(recons.shape[1]):
        temp = recons[:,i,:]
        temp2 = recons2[:,:,i]

        temp_layer = np.zeros(temp.shape)
        temp_layer2 = np.zeros(temp2.shape)

        if len(regionprops(label(temp))) > 0:
            for region in regionprops(label(temp)):
                row_c, col_c = int(region.centroid[0]), int(region.centroid[1])
                temp_layer[row_c, col_c] = 1

        if len(regionprops(label(temp2))) > 0:
            for region in regionprops(label(temp2)):
                row_c, col_c = int(region.centroid[0]), int(region.centroid[1])
                temp_layer2[row_c, col_c] = 1

        centerline[:,i,:] = temp_layer
        centerline2[:,:,i] = temp_layer2

    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, projection = '3d')
    zz, yy, xx = np.where(centerline==1) or np.where(centerline2==1)
    ax.scatter(xx,yy,zz, c = 'r', s=2)
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()





