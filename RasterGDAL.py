# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 00:15:30 2022

@author: Istel (jaeyeon)
"""

#%% Necessary Packages
import os
import numpy as np
import glob
from osgeo import gdal

#%% Define Class

# 폴더로 초기값 받고, 레스터 선택해서 array 처리하기

class RasterGDAL:
    # InputRaster의 정보 얻기
    def __init__(self, InputRaster):
        if len(os.path.splitext(InputRaster)[1]) != 0:
            Raster_Name = InputRaster
        else:
            Raster_Name = InputRaster + ".tif"
        self.Input_Raster = Raster_Name
        self.RasterOpen = gdal.Open(self.Input_Raster)
        self.band = self.RasterOpen.GetRasterBand(1)
        self.Project = self.RasterOpen.GetProjection()
        self.GeoTransform = self.RasterOpen.GetGeoTransform()
        self.nodataval = self.band.GetNoDataValue()
        self.R_Arr = self.band.ReadAsArray()


    def FitFileName(self, Inputname):
        if len(os.path.splitext(Inputname)[1]) != 0:
            Raster_Name = Inputname
        else:
            Raster_Name = Inputname + ".tif"
        return Raster_Name


    # GDAL 이용해서 레스터 작업
    def RasterToArray(self):
        nodataval = self.nodataval
        RasterArr = self.R_Arr
        return RasterArr, nodataval


    def write_geotiff(self, saveArray, OutName):
        # extent = os.path.splitext(OutRaster)[1]
        if len(os.path.splitext(OutName)[1]) != 0:
            Output_Name = OutName
        else:
            Output_Name = OutName + ".tif"
        # arr_type
        if saveArray.dtype == np.float64:
            arr_type = gdal.GDT_Float64
        else:
            arr_type = gdal.GDT_Int32
        # WriteRaster
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(Output_Name, saveArray.shape[1], saveArray.shape[0], 1, arr_type)
        out_ds.SetProjection(self.Project)
        out_ds.SetGeoTransform(self.GeoTransform)
        band = out_ds.GetRasterBand(1)
        # NodataVal = band.GetNoDataValue()
        band.SetNoDataValue(-9999)
        band.WriteArray(saveArray)
        band.FlushCache()
        band.ComputeStatistics(False)


    def ConstRaster(self, OutName):
        RasterArr = self.R_Arr
        if len(os.path.splitext(OutName)[1]) != 0:
            Output_Name = OutName
        else:
            Output_Name = OutName + ".tif"
        ConstArry = np.ones(RasterArr.shape)
        self.write_geotiff(ConstArry, Output_Name)
        return ConstArry


    def StackRaster(self, FloderPath, search_criteria):
        dirpath = FloderPath
        search_Con = search_criteria
        queryExpression = os.path.join(dirpath, search_Con)
        ListInput = glob.glob(queryExpression)
        Setlist = list(set(ListInput))
        SortedList = np.sort(Setlist).tolist()
        # Stack
        SatckIndex = np.zeros((len(SortedList), self.R_Arr.shape[0], self.R_Arr.shape[1]))
        # SatckIndex[0]
        for i, j in zip(range(0, len(SortedList)) , SortedList ):
            print(i, j)
            A = RasterGDAL(j)
            A_arr = A.RasterToArray()[0]
            arrNodata = A.RasterToArray()[1]
            NoTnullArr = np.where(A_arr == arrNodata, 0, A_arr )
            NoTnullArr.astype('f4')
            SatckIndex[i] = NoTnullArr
        return SatckIndex, SortedList

#%% Example