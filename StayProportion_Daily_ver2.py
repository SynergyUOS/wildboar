
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 01:46:58 2022

@author: system888
"""
#%% lists of used package
import os
import numpy as np

from .RasterGDAL import RasterGDAL
# feeding Raster
from .EnergyOaktree import EnergyOaktree


#%% Define Module Fuction

class StayProp_Daily:
    def __init__(self, InputDEM, Working_Dir):
        self.dirpath = Working_Dir
        os.chdir(self.dirpath)
        self.InputDEM = InputDEM
        self.baseRaster = RasterGDAL(self.InputDEM)
        self.baseRasterArr, self.noDataVal = self.baseRaster.RasterToArray()
        print("init_done")

    #### Part. Oak_Tree
    def ReadAcorn(self, Input_referTable, PropArr):
        ReadClassAcorn = EnergyOaktree(self.InputDEM, Input_referTable, PropArr)
        AcornEnergy = ReadClassAcorn.ProductionPerUnitscale()
        results = AcornEnergy
        return results

    #### Part. Grass


    #### Part. Total Stack Raster
    def readFeedRasters(self):
        ReadEnvs = RasterGDAL(self.InputDEM)
        FeedRastersPath = os.path.join(self.dirpath, "EnergyRaster")
        # print(FeedRastersPath)
        stackEnergy = ReadEnvs.StackRaster(FeedRastersPath, "*.tif")[0]
        self.KcalRaster = np.sum( stackEnergy, axis = 0 )
        return self.KcalRaster

    #### Calulate Proportion Of Stay
    def ProportionStay(self, necessaryKCal):
        # read Raster
        Read_BaseRaster = RasterGDAL(self.InputDEM)
        self.BaseArr, self.BASEnoDataVal = Read_BaseRaster.RasterToArray()

        NeedKCal = necessaryKCal
        ConstRasterNeedEnergy = np.where( self.BaseArr != self.BASEnoDataVal, NeedKCal, -9999 )

        # Read Feeding Raster
        Kcal_Raster = self.KcalRaster

        # Calculate Restults
        # 비율: 단위면적당 에너지 생산량 / 멧돼지의 1일 필요 에너지량 )
        Result = Kcal_Raster / ConstRasterNeedEnergy
        StayDailyProp = np.where( self.BaseArr != self.BASEnoDataVal, Result, -9999)

        # Write Results Raster To tif
        # write_geotiff(Output, ReProduction, DEMBand)
        os.makedirs(r"results", exist_ok = True)
        Read_BaseRaster.write_geotiff(StayDailyProp, r"results/ProportionOfStay.tif")
        return StayDailyProp


#%% Main Parameters
# dirpath = r"E:\Dropbox\60_Python_Study\99_UtilityCode\WildBoar\Temp"
# os.chdir(dirpath)


#### Input
## Input_DEM 데이터 파일
# Input_DEM = r"data/Clip2DEM.tif" # Input_DEM = "Clip2DEM.tif"

## Refer research csv
# 참고 문헌의 높이에 따른 도토리 낙하량 데이터
# Input_referTable = r"data/ProductionByDEM2.csv"