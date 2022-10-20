# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:05:32 2022

@author: univSEOULGIS
"""
#%% lists of used package
import os
import numpy as np

from .RasterGDAL import RasterGDAL
from .StayProportion_Daily_ver2 import StayProp_Daily
from .FeedAREAbyEachCell_ver2 import MinmumDistOneDay

#%%
# 향후 class로 변화 시키기
# 초기 세팅하는 부분과 자료가 다 있을때 돌리는거로 나눠서 함수 정의

def runCore( inputPath, Input_DEM, Input_PropTIF, NeedKcal ):
    # Read Input
    dirpath = inputPath
    os.chdir(dirpath)
    Input_referTable = os.path.dirname(__file__) + r"\RefDATA\ProductionByDEM2.csv"
    DEMread = RasterGDAL(Input_DEM)
    DEMarr, DEM_Nodata = DEMread.RasterToArray()
    print("Done Read MainParaMeters")

    # Read Feeding
    ConProp = RasterGDAL(Input_PropTIF)
    ConPropArr, arr_Nodata = ConProp.RasterToArray()
    ConPropArr = np.where( ConPropArr != -9999, ConPropArr, -9999)
    DEMread.write_geotiff( ConPropArr, os.path.dirname(Input_PropTIF) + r"\ConPropArr.tif")

    # 먹이자원의 에너지 계산
    # 도토리류 에너지 계산
    # inputDEM , Path
    Test_Acorn = StayProp_Daily( Input_DEM, dirpath )

    # Input_referTable, ProportionRaster
    Test_Acorn.ReadAcorn( Input_referTable, ConPropArr)
    Test_Acorn.readFeedRasters()

    print( "Run Main")
    # 멧돼지의 하루 섭취 칼로리를 42,000 kcal(150kg 개체 기준)
    Result_propStay = Test_Acorn.ProportionStay( NeedKcal )
    # 1일 에너지량 만족 최소 영역 계산
    # 최대거리 8km 셀 사이즈 30m 270
    Cellsize = 30
    MaxDistCount = int( 8000 / Cellsize) #
    Test_ClassArea = MinmumDistOneDay(Result_propStay, 270 ) # 향후 270을 MaxDistCount로 변경할 예정임
    Test_ClassArea.FeedDistancePerCell()


#%% Test Define
# dirpath = r"E:\Dropbox\60_Python_Study\99_UtilityCode\WildBoar\Temp"
# os.chdir(dirpath)
# #
# InDEM = r"ModelINPUT\Match_Gyonggi_Korea_re30m_DEM2.tif"
# InputConProp = r"ModelINPUT\ConPropArr_MaxEnt_Kanwon.tif"

# runCore(dirpath, InDEM, InputConProp, 42000)


#%% Test Define jinju
# dirpath = r"E:\Dropbox\60_Python_Study\99_UtilityCode\WildBoar\Temp"
# os.chdir(dirpath)
# #
# InDEM = r"ModelINPUT\Match_KoreaChina_1arcDEM2.tif"
# InputConProp = r"Jinju_KOFTR31_PA1_Full_MAXENT.tif"

# runCore(dirpath, InDEM, InputConProp, 42000)

