# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:41:34 2022

@author: univSEOULGIS
"""
#%% lists of used package 
import os 
import numpy as np
import pandas as pd

from RasterArcpy import RasterArcpy 

from sklearn import preprocessing
from scipy import stats


#%% Test
dirpath = r"E:\Dropbox\03.Research_Projects\2022_시나리오 기반 멧돼지 분포 및 서식 연결성 모형개발"
os.chdir(dirpath)

#### Input 
## Input_DEM 데이터 파일
Input_MCP = r"MCP_Merge.csv" # Input_DEM = "Clip2DEM.tif"
Input_Model = r"buffer_Merge.csv" # Input_DEM = "Clip2DEM.tif"

#%% joinplot
import seaborn as sns

Read_MCP = pd.read_csv( Input_MCP)
Read_Model = pd.read_csv( Input_Model) 

sub_MCP = Read_MCP[["Date", "Shape_Area"]]
sub_Model = Read_Model[["Timestamp", "Shape_Area"]]

# sub_MCP[["Type"]] = "MCP"
# sub_Model[["Type"]] = "Model"

sub_Model.rename(columns = {'Shape_Area':'B_Area'}, inplace=True)
Full_pd = pd.concat( [sub_MCP, sub_Model], axis=1)

graph = sns.jointplot(data=Full_pd, x = Full_pd["Shape_Area"], y = Full_pd["B_Area"], kind="reg")
r, p = stats.pearsonr(Full_pd["Shape_Area"], Full_pd["B_Area"])
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



