# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:36:49 2022

@author: univSEOULGIS
"""
#%% lists of used package
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from .RasterGDAL import RasterGDAL
from .sigmoidplus import sigmoidplus
#%% Main Parameters
# dirpath = r"E:\Dropbox\60_Python_Study\99_UtilityCode\WildBoar\Temp\data"
# # r"D:\wildboarDATA"
# # E:\forTest
# # E:\forTest
# # E:\Dropbox\03.Research_Projects\2021_국립생물자원관\90_DATA\DEM
# os.chdir(dirpath)

# #### Input
# ## Input_DEM 데이터 파일
# Input_DEM = "Clip2DEM.tif"
# ## Refer research csv
# # 높이에 따른 도토리 낙하량 데이터
# Input_referTable = "ProductionByDEM2.csv"

#%% define Class
class EnergyOaktree:
    def __init__(self, InputDEM, InputReferTable, PropArr):
        # Eng: Read DEM Data, # Kor: 읽어오기
        # Read_DEM
        if str(type(InputDEM)) == "<class 'numpy.ndarray'>":
            self.DEMarr = InputDEM
            self.DEMnoDataVal = -9999
        else:
            # Read raster
            self.baseRaster = RasterGDAL(InputDEM)
            self.DEMarr, self.DEMnoDataVal = self.baseRaster.RasterToArray()
        # Eng: Read refer data # Kor: 참고 자료 읽어오기
        Referdata = pd.read_csv(InputReferTable, header=0)
        self.npReferdata = Referdata.to_numpy()
        self.PropArr = PropArr
        print("init_EnergyOaktree")


    def ProductionPerUnitscale(self):
        ### Prdicted by sigmoid form
        ## RefDATA Predict
        FitRefDATA = sigmoidplus(self.npReferdata[:,0], 600, 900, 0.009)
        ## model check: R2
        NpFitChecke = np.concatenate((self.npReferdata, FitRefDATA.reshape(FitRefDATA.size,1)), axis=1)
        MeanX_ob = np.mean(NpFitChecke[:,1])
        SSE_pred = (NpFitChecke[:,2]-MeanX_ob)**2
        SST_Ob = (NpFitChecke[:,1]-MeanX_ob)**2
        self.R_quared = np.sum(SSE_pred) / np.sum(SST_Ob)
        # plot graph
        # plt.figure(figsize=(10,10))
        # plt.plot(self.npReferdata[:,0])                
        

        ## sigmoidplus (x, Ax1, Ay2, cProp)
        Pred_BySigmoid = sigmoidplus(self.DEMarr, 600, 900, 0.009)
        # 3.Mask by NoDATA
        AppNodata = np.where( self.DEMarr != self.DEMnoDataVal , Pred_BySigmoid, 0 )
        ReProduction = AppNodata * self.PropArr
        # 4.단위면적당 에너지 생산량
        # g생산량 * 단위변환 * 3.87 / 154(도토리 조사기)
        KcalPerUnitCell = np.where( self.DEMarr != self.DEMnoDataVal, ((ReProduction * 6 * 3.87 )/154) , 0)  # * 0.82

        # 4. Write PredictResults
        # Write_Raster.by Geotiff
        # write_geotiff(Output, ReProduction, DEMBand)
        os.makedirs(r"EnergyRaster", exist_ok = True)
        self.baseRaster.write_geotiff(KcalPerUnitCell, "EnergyRaster/KcalPerCell_Acorn.tif")
        return KcalPerUnitCell

#%% RunTEST
# Test = EnergyOaktree(Input_DEM, Input_referTable)
# Test2 = Test.ProductionPerUnitscale()