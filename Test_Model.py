'''
Filename: d:\70_PyCode\Wildboar\Test_Model.py
Path: d:\70_PyCode\Wildboar
Created Date: Sunday, October 23rd 2022, 11:38:32 pm
Author: Istel90

Copyright (c) 2022 Lab.Spatial data Science & Planning in University of Seoul
'''

#%% lists of used package 
import os 
import numpy as np
import pandas as pd

# from RasterArcpy import RasterArcpy 

from sklearn import preprocessing
from scipy import stats

import sys
sys.path.append(r"D:\70_PyCode")
from Wildboar.core import runCore
#%% Test
dirpath = r"E:\Dropbox\03.Research_Projects\2022_시나리오 기반 멧돼지 분포 및 서식 연결성 모형개발"
os.chdir(dirpath)

#### Input 
## Input_DEM 데이터 파일
Input_MCP = r"MCP_Merge.csv" # Input_DEM = "Clip2DEM.tif"
Input_Model = r"buffer_Merge.csv" # Input_DEM = "Clip2DEM.tif"


#%% Model Test
dirpath = r"E:\Dropbox\60_Python_Study\99_UtilityCode\임시\WildBoar\Temp"
os.chdir(dirpath)
#
InDEM = r"ModelINPUT\Match_KoreaChina_1arcDEM2.tif"
InputConProp = r"Jinju_KOFTR31_PA1_Full_MAXENT.tif"

runCore(dirpath, InDEM, InputConProp, 42000)




#%% joinplot
import seaborn as sns

Read_MCP = pd.read_csv( Input_MCP)
Read_Model = pd.read_csv( Input_Model) 

sub_MCP = Read_MCP[["Date", "Shape_Area"]]
sub_Model = Read_Model[["Timestamp", "Shape_Area"]]

sub_MCP[["Type"]] = "MCP"
sub_Model[["Type"]] = "Model"

sub_Model.rename(columns = {'Shape_Area':'B_Area'}, inplace=True)
Full_pd = pd.concat( [sub_MCP, sub_Model], axis=1)

graph = sns.jointplot(data=Full_pd, x = Full_pd["Shape_Area"], y = Full_pd["B_Area"], kind="reg")
r, p = stats.pearsonr(Full_pd["Shape_Area"], Full_pd["B_Area"])
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])

#%%
Delete_max_pd = Full_pd.drop( index=1, axis=0)

graph = sns.jointplot(data=Delete_max_pd, x = Delete_max_pd["Shape_Area"], y = Delete_max_pd["B_Area"], kind="reg")
r, p = stats.pearsonr(Delete_max_pd["Shape_Area"], Delete_max_pd["B_Area"])
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])



#%%
Delete_max_pd = Full_pd.drop( index=1, axis=0)

graph = sns.jointplot(data=Delete_max_pd, x = Delete_max_pd["Shape_Area"], y = Delete_max_pd["B_Area"], kind="reg")
r, p = stats.pearsonr(Delete_max_pd["Shape_Area"], Delete_max_pd["B_Area"])
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])



#%%
stats.ttest_ind(sub_MCP["Shape_Area"], sub_Model["Shape_Area"])


#%% Backup
scaler = preprocessing.StandardScaler().fit(sub_MCP[["Shape_Area"]])
MCP_scaled = scaler.transform(sub_MCP[["Shape_Area"]])
scaler2 = preprocessing.StandardScaler().fit(sub_Model[["Shape_Area"]])
Model_scaled = scaler2.transform(sub_Model[["Shape_Area"]])

sub_MCP[["Shape_Area"]] = MCP_scaled
sub_Model[["Shape_Area"]] = Model_scaled


phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)



