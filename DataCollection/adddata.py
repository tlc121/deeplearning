#-*-coding:utf-8-*-
import os
import pandas as pd

if __name__ == '__main__':
    path = '/Users/xavier0121/Desktop/work/xiongke_info.csv'
    df = pd.read_csv(path)
    print str(df.iloc[1]['path'])