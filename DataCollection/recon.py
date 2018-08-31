#-*-coding:utf-8-*-
import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

def copy():
    path = '/Users/xavier0121/Desktop/work/patientinfo/wholedata(1).xlsx'
    df = pd.read_excel(path)
    length = len(df)
    filepath = df[[u'路径', u'编号']].iloc[1:length]
    hardware = 'Seagate Backup plus Driver/fxxk/'+ u'复星医药' + '/'
    newpath = hardware + 'temp/'

    for i in range(0, length - 1):
        parent_path = os.path.dirname(hardware + filepath.iloc[i][u'路径'])
        for file in os.listdir(parent_path):
            if os.path.isdir(parent_path + file):
                shutil.copyfile(parent_path + file, newpath + file)

def rename():
    datapath = '/Users/xavier0121/Desktop/work/patientinfo/wholedata(1).xlsx'
    df = pd.read_excel(datapath)
    length = len(df)
    filepath = df[[u'路径', u'编号']].iloc[1:length]
    path = '/Users/xavier0121/Desktop/work/patientinfo/'
    list = os.listdir(path)
    hardware = ''
    map = {}
    count = 0

    for i in range(0, length - 1):
        original_path = hardware + str(filepath.iloc[i][u'路径'])
        parent_path = os.path.dirname(original_path)
        num = str(filepath.iloc[i][u'编号'])
        if map.has_key(num):
            map[num] += 1
        else:
            map[num] = 1
            for file in os.listdir(path + list[count]):
                if file.endswith('xlsx'):
                    oldpath = path + list[count] + '/' + file
                    shutil.copyfile(path + list[count] + '/' + file, original_path + '/' + file)
            os.rename(original_path, parent_path + list[count])
            count += 1


if __name__ == '__main__':
    path = 'Seagate Backup Plus Drive/'
    list = os.listdir(path)
    print list


