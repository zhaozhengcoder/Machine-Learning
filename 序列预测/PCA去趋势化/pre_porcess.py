import pandas as pd 
from datetime import datetime 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import pca
import pickle

def read_excel(filepath):
    df = pd.read_excel(filepath,skip_footer=1)
    df.drop_duplicates('last-update-time','first',inplace=True)
    # todo
    df.index = df['last-update-time']
    return df 

# 把数据换成一天时间的点
def generate_data_byday(df,day,begin_hour=0,end_hour=24):
    newdf = pd.DataFrame(columns=['road','vol','speed','last-update-time'])
    name = df['road'][0]
    date = '2012-11-'+str(day)
    for hour in range(begin_hour,end_hour):
        for minute in range(0,60):
            vol_item = np.nan
            speed_item = np.nan
            select_str = date+" "+str(hour)+":"+str(minute)
            if select_str in df.index:
                vol_item = df[select_str]['vol'][0]
                speed_item = df[select_str]['speed'][0]
            newdf.loc[newdf.shape[0]]=[name,vol_item,speed_item,select_str]
    return newdf

# 数据是0的点的占比
def miss_rate(data,colname='speed'):
    df = pd.isnull(data[colname])
    df_list = df.tolist()
    miss_rate = sum(df_list)/float(len(df_list))
    print ("col : ",colname,", miss rate is : ",miss_rate)
    #return miss_rate

def fill_df(result_df):
    speed_col_index = find_col_index(result_df[0],'speed')
    vol_col_index = find_col_index(result_df[0],'vol')

    for df_index in range(len(result_df)):
        for i in range(result_df[df_index].shape[0]):
            if np.isnan(result_df[df_index].iloc[i,speed_col_index]):
                if df_index ==0:  #从后面找
                    find_index=df_index+1
                    while find_index < len(result_df):
                        if np.isnan(result_df[find_index].iloc[i,speed_col_index])==False:
                            result_df[df_index].iloc[i,speed_col_index] = result_df[find_index].iloc[i,speed_col_index]
                            result_df[df_index].iloc[i,vol_col_index]   = result_df[find_index].iloc[i,vol_col_index]
                            break
                        find_index+=1
                else:   #从前面找
                    result_df[df_index].iloc[i,speed_col_index] = result_df[df_index-1].iloc[i,speed_col_index]
                    result_df[df_index].iloc[i,vol_col_index]   = result_df[df_index-1].iloc[i,vol_col_index]

def default_fill(result_df):
    for i in range(len(result_df)):
        result_df[i]=result_df[i].fillna(method='ffill')
    for i in range(len(result_df)):
        result_df[i]=result_df[i].fillna(method='bfill')
    return result_df

def df_filter(dfs):
    for df in dfs:
        df['speed']=df['speed'].apply(lambda x : min(x,110))
        df['speed']=df['speed'].apply(lambda x : max(x,10))

def find_col_index(df,columns_name):
    for i in range(len(df.columns)):
        if df.columns[i]==columns_name:
            return i 
    sys.exit("sorry, find_col_index can't find correct colnums_name .")

#原始的data里面的数据格式是dataframe，arr改成了里面也是list
def transfer(data):
    vol_col_index = 1 # 找到流量对应的列
    height = len(data)
    width = data[0].shape[0]
    arr = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            arr[i,j]=data[i].iloc[j,vol_col_index]
    return arr

# 将dfs聚合
def merge_dfs(dfs,merge_step=3):
    begin = 0
    end = int(dfs[0].shape[0]/3)
    ret_dfs=[]
    for df in dfs:
        ret_df = pd.DataFrame(columns=['road','vol','speed','last-update-time'])
        for step in range(begin,end):
            vol_item = df.iloc[step*merge_step:(step+1)*merge_step]['vol'].mean()
            speed_item =  df.iloc[step*merge_step:(step+1)*merge_step]['speed'].mean()
            name = df.iloc[step*merge_step]['road']
            time = df.iloc[step*merge_step]['last-update-time']
            ret_df.loc[ret_df.shape[0]]=[name,vol_item,speed_item,time]
        ret_dfs.append(ret_df)
    print ("ori dfs shape is : ",dfs[0].shape)
    print ("ret dfs shape is : ",ret_dfs[0].shape)
    return ret_dfs


def pre_process():
    abspath='C:/Users/wwwa8/Documents/GitHub/Machine-Learning/序列预测/PCA去趋势化/7.xls'
    #filepath='7.xls'
    df = read_excel(abspath)
    # 起始日期
    days=range(9,13)
    # 每一天每一分钟对应一个点的格式
    dfs=[]
    begin_hour = 0
    end_hour = 24
    for day in days:
        dfs.append(generate_data_byday(df,day,begin_hour,end_hour))
    for df in dfs:
        miss_rate(df)
    fill_df(dfs)
    for df in dfs:
        miss_rate(df)
    #填充nan
    dfs = default_fill(dfs)
    #过滤异常值，特别大的，特别小的
    df_filter(dfs)
    # 按分钟聚合
    dfs = merge_dfs(dfs,merge_step=3)
    arr = transfer(dfs)
    return arr


def mypickle(filepath,data):
    abs_path='C:/Users/wwwa8/Documents/GitHub/Machine-Learning/序列预测/PCA去趋势化/'
    f=open(abs_path+filepath,'wb')
    pickle.dump(data,f)
    f.close()

if __name__ =="__main__":
    arr = pre_process()
    print ("shape : ",arr.shape)
    pca_obj=pca.PCA(arr,2)
    #print (pca_obj.main_x.shape)
    #print (pca_obj.rest_x.shape)

    mypickle('dump_arr_9-13.txt',arr)
    mypickle('dump_main_x_9-13.txt',pca_obj.main_x)
    mypickle('dump_rest_x_9-13.txt',pca_obj.rest_x)