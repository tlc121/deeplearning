#-*-coding:utf-8-*-
import numpy as np
import SimpleITK as sitk

def ReadDICOMFolder(folderName):
    reader=sitk.ImageSeriesReader()
    seriesIDs=reader.GetGDCMSeriesIDs(folderName)
    for idx in seriesIDs:
        try:
            dicomfilenames=reader.GetGDCMSeriesFileNames(folderName,idx)
            reader.SetFileNames(dicomfilenames)
            image=reader.Execute()
            imageRaw=sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            imageRaw=np.flip(imageRaw,0)
        except:
            pass
    return imageRaw, spacing


ct,spacing = ReadDICOMFolder(u'/Users/xavier0121/Desktop/work/lc/付永龙2017.8.24/DICOM/1935562/007-0746-101/')
print ct.shape