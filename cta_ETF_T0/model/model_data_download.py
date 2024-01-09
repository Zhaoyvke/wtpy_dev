# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:12:14 2023

@author: OptionTeam
"""

import paramiko
import scp

try:
    print('模型数据开始下载')
    host = "222.73.246.137" #数据库
    # host = "222.73.246.135" #覃老师数据库
    
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host,port=22,username="lixy",password='wuzhi')
    # ssh.connect(hostname=host,port=22,username="lixy",password='WZ!@#$%^')
    scp1 = scp.SCPClient(ssh.get_transport())
    
    scp1.get('/home/lixy/T0/T0_level1_volume_fix_pred_model.pkl','./ETF_T0/model/')
    scp1.get('/home/lixy/T0/T0_level1_volume_fix_pred_scaler.pkl','./ETF_T0/model/')
    scp1.get('/home/lixy/T0/T0_level1_volume_rolling_pred_model.pkl','./ETF_T0/model/')
    scp1.get('/home/lixy/T0/T0_level1_volume_rolling_pred_scaler.pkl','./ETF_T0/model/')
    scp1.get('/home/lixy/T0/predict_level1_volume_ETF.parquet','./ETF_T0/model/')
    input('ETF_T0模型下载完成')
except Exception as e:
    print('模型下载失败')
    print(e)
    input('error!')