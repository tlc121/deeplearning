#-*-coding:utf-8-*-
import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

def generatecount(num):
    result = ''
    if num / 10 is 0:
        result = '000' + str(num)
    elif num / 100 is 0:
        result = '00' + str(num)
    elif num / 1000 is 0:
        result = '0' + str(num)
    else:
        result = str(num)
    return result


def readdata(filepath):
    #read the whole data
    df = pd.read_csv(filepath + 'chanyi_new.csv')
    length = len(df)
    #first create the all folders for each patient
    #map is a key-value pair for {num, count}
    #map2 is for {folder. num}
    map = {}
    map2 = {}
    path = df['path']
    diameter = df['diameter']
    name = df['name']

    #确定文件夹的名字，取最大直径决定命名
    max_size = 0
    count = 0
    seris = ''
    for i in tqdm(range(0, length)):
        if map.has_key(str(path.iloc[i])):
            if max_size < float(diameter.iloc[i]):
                max_size = float(diameter.iloc[i])
                type_now = str(name.iloc[i])
            if type is not type_now:
                type = type_now
                newdir = filepath + '007-' + seris + '-' + type_now + '/'
                map2[newdir] = str(path.iloc[i])
            map[str(path.iloc[i])] += 1
        else:
            max_size = float(diameter.iloc[i])
            map[str(path.iloc[i])] = 1
            count += 1
            seris = generatecount(count)
            type = str(name.iloc[i])
            os.makedirs(filepath + '007-' + seris + '-' + type + '/')
            newdir = filepath + '007-' + seris + '-' + type + '/'
            map2[newdir] = str(path.iloc[i])
    return map, map2

def addfiles(map, map2, filepath):
    df = pd.read_csv(filepath + 'chanyi_new.csv')
    length = len(df)
    row = 0
    row1 = 0

    for folder in tqdm(os.listdir(filepath)):
        if os.path.isdir(filepath + folder):
            foldername = filepath + folder
            oldname = '/Users/xavier0121/Desktop/temp/allinfo.xlsx'
            shutil.copyfile(oldname, foldername + '/' + folder + 'info.xlsx')
            patient_path = map2[foldername+'/']
            num_jiejie = map[patient_path]
            allinfo = pd.read_excel(foldername + '/' + folder + 'info.xlsx')
            writer = pd.ExcelWriter(foldername + '/' + folder + 'info.xlsx')
            allinfo.iloc[1, 0] = folder
            allinfo.iloc[1, 9] = 0
            allinfo.iloc[1, 10] = 0
            allinfo.iloc[1, 11] = 0
            allinfo.iloc[1, 12] = 0
            allinfo.iloc[1, 13] = 0
            allinfo.iloc[1, 14] = 0
            for i in range(row, row + num_jiejie):

                allinfo.iloc[1, 1] = df.iloc[row]['dicom_file_info.SOPInstanceUID']
                allinfo.iloc[1, 2] = df.iloc[row]['dicom_file_info.InstitutionName']
                allinfo.iloc[1, 5] = df.iloc[row]['dicom_file_info.WindowWidth']
                allinfo.iloc[1, 7] = df.iloc[row]['resolution']
                allinfo.iloc[1, 8] = df.iloc[row]['dicom_file_info.SliceThickness']

                if int(df.iloc[i]['name']) is 100:
                    allinfo.iloc[1, 9] += 1
                if int(df.iloc[i]['name']) is 101:
                    allinfo.iloc[1, 10] += 1
                if int(df.iloc[i]['name']) is 200:
                    allinfo.iloc[1, 12] += 1
                if int(df.iloc[i]['name']) is 201:
                    allinfo.iloc[1, 11] += 1
                if int(df.iloc[i]['name']) is 300:
                    allinfo.ilioc[1, 14] += 1
                if int(df.iloc[i]['name']) is 301:
                    allinfo.iloc[1, 13] += 1

            row = row + num_jiejie
            allinfo.to_excel(writer, 'Sheet1')
            writer.save()

            oldname1 = '/Users/xavier0121/Desktop/temp/each.xlsx'
            #create the file for each jiejie
            max_size = 0
            for i in range(0, num_jiejie):
                if max_size < float(df.iloc[row1 + i]['diameter']):
                    max_size = float(df.iloc[row1 + i]['diameter'])
                try:
                    shutil.copyfile(oldname1, foldername + '/' + folder + 'mark-' + str(i+1) + '.xlsx')
                    writer1 = pd.ExcelWriter(foldername + '/' + folder + 'mark-' + str(i+1) + '.xlsx')
                    eachinfo = pd.read_excel(oldname1)
                    eachinfo.iloc[0, 0] = int(df.iloc[row1+i]['x'])
                    eachinfo.iloc[0, 1] = int(df.iloc[row1+i]['y'])
                    eachinfo.iloc[0, 2] = int(df.iloc[row1+i]['z'])
                    eachinfo.iloc[0, 3] = max_size
                    eachinfo.iloc[0, 5] = float(df.iloc[row1+i]['density'])
                    eachinfo.iloc[0, 7] = int(df.iloc[row1+i]['z_max_area'])
                    eachinfo.iloc[0, 6] = int(df.iloc[row1+i]['type_nodule'])

                    if str(df.iloc[row1+i]['type_nodule']) is '1':
                        eachinfo.iloc[0, 8] = str(df.iloc[row1+i]['z_max_area'])
                        eachinfo.iloc[0, 11] = int(df.iloc[row1+i]['z_end']) - int(df.iloc[row1+i]['z_init'])
                    if str(df.iloc[row1 + i]['type_nodule']) is '3':
                        eachinfo.iloc[0, 9] = str(df.iloc[row1+i]['z_max_area'])
                        eachinfo.iloc[0, 12] = int(df.iloc[row1+i]['z_end']) - int(df.iloc[row1+i]['z_init'])

                    eachinfo.iloc[0, 10] = float(df.iloc[row1+i]['z_end']) - float(df.iloc[row1+i]['z_init'])
                    eachinfo.to_excel(writer1, 'Sheet1')
                    writer1.save()
                except Exception, err:
                    print err
            row1 = row1 + num_jiejie




if __name__ == '__main__':
    filepath = '/Users/xavier0121/Desktop/work/patientinfo3/'
    map, map2 = readdata(filepath)
    addfiles(map, map2, filepath)


