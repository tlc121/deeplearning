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
    df = pd.read_excel(filepath + 'chanyi_unhealthy.csv')
    length = len(df)
    #first create the all folders for each patient
    #map is a key-value pair for {num, count}
    #map2 is for {folder. num}
    map = {}
    map2 = {}
    map3 = {}
    path = df[u'路径'].iloc[1:length]
    diameter = df[[u'结节最大径长', u'密度分类（1实性，0纯磨玻璃，0混合）', u'密度分类（0实性，1纯磨玻璃，0混合）', u'密度分类（0实性，0纯磨玻璃，1混合）']].iloc[1:length]

    #确定文件夹的名字，取最大直径决定命名
    max_size = 0
    count = 0
    seris = ''
    type = ''
    type_now = ''
    for i in tqdm(range(0, length-1)):
        if map.has_key(str(path.iloc[i])):
            if float(diameter.iloc[i][u'结节最大径长']) > max_size:
                max_size = float(diameter.iloc[i][u'结节最大径长'])
                pd.
                if str(diameter.iloc[i][u'密度分类（1实性，0纯磨玻璃，0混合）']) is '1':
                    if max_size < 6:
                        type_now = '100'
                    else:
                        type_now = '101'
                elif str(diameter.iloc[i][u'密度分类（0实性，1纯磨玻璃，0混合）']) is '1':
                    if max_size < 5:
                        type_now = '300'
                    else:
                        type_now = '301'
                elif str(diameter.iloc[i][u'密度分类（0实性，0纯磨玻璃，1混合）']) is '1':
                    if max_size < 6:
                        type_now = '200'
                    else:
                        type_now = '201'
                map[str(path.iloc[i])] += 1
                if type is not type_now:
                    os.rename(filepath + '007-' + seris + '-' + type + '/', filepath + '007-' + seris + '-' + type_now + '/')
                    type = type_now
                    newdir = filepath + '007-' + seris + '-' + type_now + '/'
                    map2[newdir] = str(path.iloc[i])
                    map3['007-' + seris + '-' + type] = realpath.iloc[i].encode('utf-8')
        else:
            max_size = float(diameter.iloc[i][u'结节最大径长'])
            map[str(path.iloc[i])] = 1
            count += 1
            seris = generatecount(count)
            if str(diameter.iloc[i][u'密度分类（1实性，0纯磨玻璃，0混合）']) is '1':
                if max_size < 6:
                    type = '100'
                else:
                    type = '101'
            elif str(diameter.iloc[i][u'密度分类（0实性，1纯磨玻璃，0混合）']) is '1':
                if max_size < 5:
                    type = '300'
                else:
                    type = '301'
            elif str(diameter.iloc[i][u'密度分类（0实性，0纯磨玻璃，1混合）']) is '1':
                if max_size < 6:
                    type = '200'
                else:
                    type = '201'
            os.makedirs(filepath + '007-' + seris + '-' + type + '/')
            newdir = filepath + '007-' + seris + '-' + type + '/'
            map2[newdir] = str(path.iloc[i])
            map3['007-' + seris + '-' + type] = realpath.iloc[i].encode('utf-8')
    return map, map2, map3

def addfiles(map, map2, map3, filepath):
    df = pd.read_excel(filepath + 'wholedata(1) copy.xlsx')
    length = len(df)
    df = df.iloc[1:length]
    row = 0
    row1 = 0
    df1 = pd.read_csv('/Users/xavier0121/Desktop/work/xiongke_info.csv')

    for folder in tqdm(os.listdir(filepath)):
        if os.path.isdir(filepath + folder):
            path = map3[folder]
            index = df1[df1.path == path].index.tolist()
            foldername = filepath + folder
            oldname = '/Users/xavier0121/Desktop/temp/allinfo.xlsx'
            shutil.copyfile(oldname, foldername + '/' + folder + 'info.xlsx')
            patient_num = map2[foldername+'/']
            num_jiejie = map[patient_num]
            allinfo = pd.read_excel(foldername + '/' + folder + 'info.xlsx')
            writer = pd.ExcelWriter(foldername + '/' + folder + 'info.xlsx')
       #      change = allinfo[[u'图像样本命名', u'实性结节（＜6mm）个数', u'实性结节(≥6mm)个数', u'部分实性结节(≥6mm)个数', u'部分实性结节（＜6mm）个数',
       # u'磨玻璃结节(≥6mm)个数', u'磨玻璃结节（＜6mm）个数', u'无明显异常（填写范例：是填写“1”/否填写“0”）']]
            allinfo.iloc[1, 0] = folder
            allinfo.iloc[1, 9] = 0
            allinfo.iloc[1, 10] = 0
            allinfo.iloc[1, 11] = 0
            allinfo.iloc[1, 12] = 0
            allinfo.iloc[1, 13] = 0
            allinfo.iloc[1, 14] = 0
            if len(index) is not 0:
                allinfo.iloc[1, 1] = df1.iloc[index[0]]['dicom_file_info.SOPInstanceUID']
                allinfo.iloc[1, 2] = df1.iloc[index[0]]['dicom_file_info.InstitutionName']
                allinfo.iloc[1, 5] = df1.iloc[index[0]]['dicom_file_info.WindowWidth']
                allinfo.iloc[1, 7] = df1.iloc[index[0]]['resolution']
                allinfo.iloc[1, 8] = df1.iloc[index[0]]['dicom_file_info.SliceThickness']
            for i in range(row, row + num_jiejie):
                allinfo.iloc[1, 3] = u'philips64排'
                if str(df.iloc[i][u'密度分类（1实性，0纯磨玻璃，0混合）']) is '1':
                    if float(df.iloc[i][u'结节最大径长']) < 6:
                        allinfo.iloc[1, 9] += 1
                    else:
                        allinfo.iloc[1, 10] += 1
                if str(df.iloc[i][u'密度分类（0实性，1纯磨玻璃，0混合）']) is '1':
                    if float(df.iloc[i][u'结节最大径长']) < 6:
                        allinfo.iloc[1, 14] += 1
                    else:
                        allinfo.iloc[1, 13] += 1
                if str(df.iloc[i][u'密度分类（0实性，0纯磨玻璃，1混合）']) is '1':
                    if float(df.iloc[i][u'结节最大径长']) < 6:
                        allinfo.iloc[1, 12] += 1
                    else:
                        allinfo.iloc[1, 11] += 1
                row = row + num_jiejie
            allinfo.to_excel(writer, 'Sheet1')
            writer.save()

            oldname1 = '/Users/xavier0121/Desktop/temp/each.xlsx'
            #create the file for each jiejie
            for i in range(0, num_jiejie):
                try:
                    shutil.copyfile(oldname1, foldername + '/' + folder + 'mark-' + str(i+1) + '.xlsx')
                    writer1 = pd.ExcelWriter(foldername + '/' + folder + 'mark-' + str(i+1) + '.xlsx')
                    eachinfo = pd.read_excel(oldname1)
                    eachinfo.iloc[0, 0] = int(df.iloc[row1+i][u'x坐标数字'])
                    eachinfo.iloc[0, 1] = int(df.iloc[row1+i][u'y坐标数字'])
                    eachinfo.iloc[0, 2] = int(df.iloc[row1+i][u'z坐标数字'])
                    eachinfo.iloc[0, 3] = float(df.iloc[row1+i][u'结节最大径长'])
                    eachinfo.iloc[0, 5] = float(df.iloc[row1+i][u'密度（-1000至1000）'])
                    eachinfo.iloc[0, 7] = int(df.iloc[row1+i][u'z坐标数字'])

                    if str(df.iloc[row1+i][u'密度分类（1实性，0纯磨玻璃，0混合）']) is '1':
                        eachinfo.iloc[0, 8] = str(df.iloc[row1+i][u'z坐标数字'])
                        if not (np.isnan(df.iloc[row1 + i][u'z结束平面']) or np.isnan(df.iloc[row1+i][u'z开始平面'])):
                            eachinfo.iloc[0, 11] = int(df.iloc[row1+i][u'z结束平面'])- int(df.iloc[row1+i][u'z开始平面'])
                    if str(df.iloc[row1 + i][u'密度分类（0实性，1纯磨玻璃，0混合）']) is '1':
                        eachinfo.iloc[0, 9] = str(df.iloc[row1+i][u'z坐标数字'])
                        if not (np.isnan(df.iloc[row1 + i][u'z结束平面']) or np.isnan(df.iloc[row1 + i][u'z开始平面'])):
                            eachinfo.iloc[0, 12] = int(df.iloc[row1+i][u'z结束平面'])- int(df.iloc[row1+i][u'z开始平面'])
                    if not(np.isnan(df.iloc[row1+i][u'z结束平面']) or np.isnan(df.iloc[row1+i][u'z开始平面'])):
                        eachinfo.iloc[0, 10] = float(df.iloc[row1+i][u'z结束平面'])- float(df.iloc[row1+i][u'z开始平面'])
                    eachinfo.to_excel(writer1, 'Sheet1')
                    writer1.save()
                    row1 = row1 + num_jiejie
                except Exception, err:
                    print err



if __name__ == '__main__':
    filepath = '/Users/xavier0121/Desktop/work/patientinfo2/'
    map, map2, map3 = readdata(filepath)
    addfiles(map, map2, map3, filepath)

