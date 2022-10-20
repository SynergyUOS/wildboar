# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:14:17 2022

@author: system888
"""

#%% package list
import numpy as np
import cv2

from .RasterGDAL import RasterGDAL
# from functools import reduce

#%% Define function
def circleKernel(x):
    radius = x
    size = 2 * radius + 1
    radiusSquarred= radius**2
    mask = np.zeros((size, size))
    distance = lambda x, y: (x-radius)**2 + (y-radius)**2
    for i in range(size):
        for j in range (2 * radius+ 1):
            if distance(i, j) <= radiusSquarred:
                mask[i, j] = 1
    return mask

#### 가중치 적용하는 방법
# Need "read_geotiff, write_geotiff"
def WeightRasterStack(WeightRasterlist):
    print("WeightRasterlist: %s" %(WeightRasterlist))
    for i in range(len(WeightRasterlist)):
        print(i)
        if i == 0 :
            ReadRaster = RasterGDAL(WeightRasterlist[i])
            RasterArr, noDataVal = ReadRaster.RasterToArray()
            Numpy_Nodata = np.where( RasterArr != noDataVal, RasterArr, 0 )
            StackSize = RasterArr.shape
            StackRaster = np.zeros((len(WeightRasterlist), StackSize[0], StackSize[1]))
            StackRaster[i] = Numpy_Nodata
        else:
            ReadRaster = RasterGDAL(WeightRasterlist[i])
            Numpy_Nodata = np.where( RasterArr != noDataVal, RasterArr, 0 )
            StackRaster[i] = Numpy_Nodata
    results = np.prod( StackRaster, axis = 0)
    return results


#%% Define class Fuction

class MinmumDistOneDay:
    def __init__(self, inputPropRaster, MaxRadius):
        # MaxRadius = 선행 연구의 멧돼지의 최대 거리
        self.MaxRadius = MaxRadius

        if str(type(inputPropRaster)) == "<class 'numpy.ndarray'>":
            self.RasterArr = inputPropRaster
            self.noDataVal = -9999
        else:
            # Read raster
            self.ReadRaster = RasterGDAL(inputPropRaster)
            self.RasterArr, self.noDataVal = self.ReadRaster.RasterToArray()
        return print("init done")


    # Need "read_geotiff, write_geotiff"
    def FeedDistancePerCell(self):
        # input
        PerDay_Nodata = np.where( self.RasterArr != self.noDataVal, self.RasterArr, 0 )
        img_size = PerDay_Nodata.shape
        radius_index = []
        for IterRadius in range(1, self.MaxRadius):
            print(f"radius = {IterRadius}")
            # kernel 정의
            Kernel = circleKernel(IterRadius)
            #외곽부 확장하기
            img_extend = np.zeros(np.array(img_size) + IterRadius*2)
            img_extend[IterRadius:-IterRadius, IterRadius:-IterRadius] = PerDay_Nodata
            # knernel 적용
            filteredImg = cv2.filter2D(img_extend, -1, Kernel)
            # 원래 이미지 크기부분의 값 추출
            RawIndex = filteredImg[IterRadius:-IterRadius, IterRadius:-IterRadius]
            # 커널이내의 합의 1(하루) 과 같거나 클때 1부여
            if IterRadius == 1:
                Con_Index = np.zeros( (len(range(1, self.MaxRadius)), PerDay_Nodata.shape[0], PerDay_Nodata.shape[1]) )
                cal = np.where( RawIndex >= 1, 1, 0)
                Con_Index[IterRadius-1] = cal
            elif IterRadius == self.MaxRadius - 1:
                cal = np.where( RawIndex >= 1, 1, RawIndex)
                Con_Index[IterRadius-1] = cal
            else:
                # 처음에 R_index를 만듬
                cal = np.where( RawIndex >= 1, 1, 0)
                Con_Index[IterRadius-1] = cal
            radius_index.append(IterRadius)
        #
        Stack_Sum = np.sum(Con_Index, axis = 0)
        # Results / (9: 반복횟수)
        MinRadius = np.where( Stack_Sum != 0, self.MaxRadius - Stack_Sum , self.noDataVal)

        # Save/Write Raster
        if str(type(self.RasterArr)) == "<class 'numpy.ndarray'>":
            self.ReadRaster = RasterGDAL(r"results\ProportionOfStay.tif")
            self.RasterArr, self.noDataVal = self.ReadRaster.RasterToArray()
            self.ReadRaster.write_geotiff(MinRadius, r"results\MinmumDistOneDay.tif" )
        else:
            self.ReadRaster.write_geotiff(MinRadius, r"results\MinmumDistOneDay.tif" )
        results = MinRadius
        return results


    # # Need "read_geotiff, write_geotiff"
    # def FeedDistancePerCell_withWeight(self, *Weightlist):
    #     # InputDATA
    #     PerDay_Nodata = np.where( self.RasterArr != self.noDataVal, self.RasterArr, 0 )
    #     img_size = PerDay_Nodata.shape

    #     # WeightRaster multiply
    #     MultiplyWeightStack = WeightRasterStack(Weightlist)
    #     # MultiplyWeightStack = np.multiply(StackRaster, 0)
    #     for IterRadius in range(1, self.MaxRadius):
    #         print("radius = %s" %(IterRadius))
    #         # kernel 정의
    #         Kernel = circleKernel(IterRadius)
    #         #외곽부 확장하기
    #         img_extend = np.zeros(np.array(img_size) + IterRadius*2)
    #         img_extend[IterRadius:-IterRadius, IterRadius:-IterRadius] = PerDay_Nodata

    #         ## 가중치 외곽부 확장하기
    #         Weight_extend = np.zeros(np.array(img_size) + IterRadius*2)
    #         Weight_extend[IterRadius:-IterRadius, IterRadius:-IterRadius] = MultiplyWeightStack
    #         # filteredWeight = cv2.filter2D(Weight_extend, -1, Kernel)
    #         # WeIndex = filteredWeight[IterRadius:-IterRadius, IterRadius:-IterRadius]

    #         # 가중치 원 데이터에 적용하기
    #         Weighted_extended = img_extend * Weight_extend

    #         # knernel 적용
    #         filteredImg = cv2.filter2D(Weighted_extended, -1, Kernel)
    #         # 원래 이미지 크기부분의 값 추출
    #         ConArrayBystep = filteredImg[IterRadius:-IterRadius, IterRadius:-IterRadius]

    #         # Weighted_P_index
    #         # 커널이내의 합의 1(하루) 과 같거나 클때 1부여
    #         if IterRadius == 1:
    #             results_stack = np.zeros( (len(range(1, self.MaxRadius)), PerDay_Nodata.shape[0], PerDay_Nodata.shape[1])    )
    #             cal = np.where( ConArrayBystep >= 1, 1, 0)
    #             results_stack[IterRadius-1] = cal
    #         else:
    #             cal = np.where( ConArrayBystep >= 1, 1, 0)
    #             results_stack[IterRadius-1] = cal
    #     #
    #     Stack_Sum = np.sum(results_stack, axis = 0)
    #     # Results / (9: 반복횟수)
    #     MinRadius = np.where( Stack_Sum != 0, self.MaxRadius - Stack_Sum , self.noDataVal)

    #     # Save/Write Raster
    #     if str(type(self.RasterArr)) == "<class 'numpy.ndarray'>":
    #         self.ReadRaster = RasterGDAL(r"results\ProportionOfStay.tif")
    #         self.RasterArr, self.noDataVal = self.ReadRaster.RasterToArray()
    #         self.ReadRaster.write_geotiff(MinRadius, r"results\MinmumDistOneDay.tif" )
    #     else:
    #         self.ReadRaster.write_geotiff(MinRadius, r"results\MinmumDistOneDay.tif" )
    #     results = MinRadius

    #     return results


#%% def TEST
# circleKernel(2)

# FeedDistancePerCell(Input_PerDay, 9)

# Result = FeedDistancePerCell_withWeight( Input_PerDay, 9, InputModifiedRF)

# A = np.array([[1,1],[1,1]])
# B = np.array([[-9,2],[3,4]])
# C = np.multiply(A, B)
# D = A*B
