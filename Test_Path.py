'''
Filename: d:\70_PyCode\Wildboar\Test_Path.py
Path: d:\70_PyCode\Wildboar
Created Date: Tuesday, September 5th 2023, 2:39:47 pm
Author: Istel

Copyright (c) 2023 Lab.Synergy
'''
#%%
import os 
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\70_PyCode\TempPythonCode")
sys.path.append(r"D:\70_PyCode")

from RasterGDAL import RasterGDAL

#%%
# 종료기준1: 먹이를 다 먹었다
Con_Feeding = 1 # Propotion

# 종료기준2: 최대이동거리만큼 이동했다
Con_Distance = 15000 # m
# cell size: 10m


# 기준 2개 중 하나라도 만족하지 못하면 종료
Cell_Prop = 0.05
cell_size = 30
while (Con_Feeding > 0) and (Con_Distance > 0):
    Con_Feeding = round(Con_Feeding - Cell_Prop, 5)
    Con_Distance = Con_Distance - cell_size    
    print("먹이 만족도", Con_Feeding)
    print("최대이동거리", Con_Distance)


# 이동경로도출
# 이동경로 좌표의 먹이만족율 취득
RawCoord = np.array([0,0])
# 좌표이동
MoveCursor = np.array([2,3])
MoveResult = RawCoord + MoveCursor
MoveResult

#%%
# 주변 8개 셀의 상대좌표 데이터
setRefIndex = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1],           [0, 1],
                        [1, -1], [1, 0],  [1, 1]])
setRef_dist = np.array([1.414, 1, 1.414,
                        1,          1,
                        1.414, 1, 1.414])
# 저항, 전류의 개념으로, 저항이 높을수록 전류가 흐르지 않는다. 이동이 어려워 선택되지 않는다. 
def ToSelectProp(SurroundProp):
    resistances = SurroundProp
    # 각 저항의 저항값 계산 (역수)
    inverse_resistances = 1 / resistances
    # 전류 비율 계산
    total_inverse_resistance = np.sum(inverse_resistances)
    current_ratios = inverse_resistances / total_inverse_resistance
    return current_ratios

#### Sample Data
# 이동 난이도를 예시로 함
Sample_CostMap = np.linspace(1, 10, 25).reshape(5,5)
Sample_PropMap = np.linspace(0.1, 1, 25).reshape(5,5)

#
StartCoord = np.array([2,3])
MoveResult = StartCoord


#### Main Part test
# 시작점 주변의 8개 셀의 좌표 취득
surrounding_indices = StartCoord + setRefIndex
# 주변 8개의 이동난이도 취득
getProp_inGrid = Sample_CostMap[surrounding_indices[:,0], surrounding_indices[:,1]]
# 건물이나 물이 있는 경우, 이동할 수 없는 경우 9999로 처리 예시
getProp_inGrid[2] = 9999
# 각 8방향에 대한 이동의 흐름의 비율 계산
CellSelectProp = ToSelectProp(getProp_inGrid)  
GetRandomSurroundCell = np.random.choice(np.arange(0, 8), 1, p=CellSelectProp)
print("이동방향", GetRandomSurroundCell)
# 역순으로 이동된 결과의 좌표와 먹이자원 만족비율 취득
Selected_PathCell_Index = surrounding_indices[GetRandomSurroundCell].reshape(2)
Selected_StepCell = Selected_PathCell_Index
Sample_PropMap[Selected_PathCell_Index[0]][Selected_PathCell_Index[1]]
Get_MovedDist = setRef_dist[GetRandomSurroundCell]
print("이동거리", Get_MovedDist)

# 2 step
MoveResult = np.vstack((MoveResult, Selected_StepCell))
MovedStartCoord = MoveResult[-1]



#%%
#### Sample Data
# 이동 난이도를 예시로 함
Sample_CostMap = np.linspace(1, 10, 25).reshape(5,5)
Sample_PropMap = np.linspace(0.1, 1, 25).reshape(5,5)

getExtend_X = Sample_CostMap.shape[0]
getExtend_Y = Sample_CostMap.shape[1]


MovedStartCoord = np.array([2, 3])


# 종료기준1: 먹이를 다 먹었다
Con_Feeding = 1 # Propotion

# 종료기준2: 최대이동거리만큼 이동했다
Con_Distance = 5 # m
# cell size: 10m

MoveResult = MovedStartCoord
while (Con_Feeding > 0) and (Con_Distance > 0):
    surrounding_indices = MovedStartCoord + setRefIndex
    # 범위에 맞춰서, 외곽부분 데이터 수정
    X_coordList = surrounding_indices[:,0]
    Y_coordList = surrounding_indices[:,1]
    X_coordList[X_coordList < 0] = 0
    X_coordList[X_coordList > (getExtend_X - 1)] = (getExtend_X - 1)
    Y_coordList[Y_coordList < 0] = 0
    Y_coordList[Y_coordList > (getExtend_Y - 1)] = (getExtend_Y - 1)
    surrounding_indices[:,0] = X_coordList
    surrounding_indices[:,1] = Y_coordList
    # print(surrounding_indices)
    # 주변 8개의 이동난이도 취득
    getProp_inGrid = Sample_CostMap[surrounding_indices[:,0], surrounding_indices[:,1]]
    # 각 8방향에 대한 이동의 흐름의 비율 계산
    CellSelectProp = ToSelectProp(getProp_inGrid)  
    GetRandomSurroundCell = np.random.choice(np.arange(0, 8), 1, p=CellSelectProp)
    # print("이동방향", GetRandomSurroundCell)
    # 역순으로 이동된 결과의 좌표와 먹이자원 만족비율 취득
    Selected_PathCell_Index = surrounding_indices[GetRandomSurroundCell].reshape(2)
    Selected_Coord = Selected_PathCell_Index
    # 좌표, 만족비율, 이동거리
    Get_FeedingProp = Sample_PropMap[Selected_PathCell_Index[0]][Selected_PathCell_Index[1]] # 먹이자원 만족비율 취득
    Get_MovedDist = setRef_dist[GetRandomSurroundCell] # 이동거리 취득
    # 이동된 셀의 좌표 저장
    MoveResult = np.vstack((MoveResult, Selected_Coord))
    # Set Next Start Coord
    MovedStartCoord = MoveResult[-1]
    
    # 종료기준 계산
    Con_Feeding = round(Con_Feeding - Get_FeedingProp, 5)
    Con_Distance = Con_Distance - Get_MovedDist    
    print("먹이 만족도", Con_Feeding)
    print("최대이동거리", Con_Distance)
    
    if (Con_Feeding <= 0) | (Con_Distance <= 0):
        if Con_Feeding <= 0:
            print("먹이를 다 먹었다")
        elif Con_Distance <= 0:
            print("최대이동거리를 다 돌았다")
        else:
            print("Error")

print(MoveResult)


#%% 완료된 함수정의
def ToSelectProp(SurroundProp):
    resistances = SurroundProp
    # 각 저항의 저항값 계산 (역수)
    inverse_resistances = 1 / resistances
    # 전류 비율 계산
    total_inverse_resistance = np.sum(inverse_resistances)
    current_ratios = inverse_resistances / total_inverse_resistance
    return current_ratios



def FeedingPropPath(StartCoord, CostMap, FeedingPropMap, MaxDistance = 1500, cell_size = 10):
    # 종료기준1: 먹이를 다 먹었다
    Con_Feeding = 1 # Propotion
    # 종료기준2: 최대이동거리만큼 이동했다
    Con_Distance = MaxDistance # m
    # 주변 8개 셀의 상대좌표 데이터
    setRefIndex = np.array([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1],           [0, 1],
                            [1, -1], [1, 0],  [1, 1]])

    setRef_dist = np.array([1.414, 1, 1.414,
                            1,          1,
                            1.414, 1, 1.414]) * cell_size
    getExtend_X = CostMap.shape[0]
    getExtend_Y = CostMap.shape[1]
    #### Main Part 
    MovedStartCoord = StartCoord
    MoveResult = MovedStartCoord
    # print("While 시작")
    # 반복문
    while (Con_Feeding > 0) and (Con_Distance > 0):
        surrounding_indices = MovedStartCoord + setRefIndex
        # 범위에 맞춰서, 외곽부분 데이터 수정
        X_coordList = surrounding_indices[:,0]
        Y_coordList = surrounding_indices[:,1]
        X_coordList[X_coordList < 0] = 0
        X_coordList[X_coordList > (getExtend_X - 1)] = (getExtend_X - 1)
        Y_coordList[Y_coordList < 0] = 0
        Y_coordList[Y_coordList > (getExtend_Y - 1)] = (getExtend_Y - 1)
        surrounding_indices[:,0] = X_coordList
        surrounding_indices[:,1] = Y_coordList
        # print(surrounding_indices)
        # 주변 8개의 이동난이도 취득
        getProp_inGrid = CostMap[surrounding_indices[:,0], surrounding_indices[:,1]]
        # 각 8방향에 대한 이동의 흐름의 비율 계산
        CellSelectProp = ToSelectProp(getProp_inGrid)  
        GetRandomSurroundCell = np.random.choice(np.arange(0, 8), 1, p=CellSelectProp)
        # print("이동방향", GetRandomSurroundCell)
        # 역순으로 이동된 결과의 좌표와 먹이자원 만족비율 취득
        Selected_PathCell_Index = surrounding_indices[GetRandomSurroundCell].reshape(2)
        Selected_Coord = Selected_PathCell_Index
        # 좌표, 만족비율, 이동거리
        Get_FeedingProp = FeedingPropMap[Selected_PathCell_Index[0]][Selected_PathCell_Index[1]] # 먹이자원 만족비율 취득
        Get_MovedDist = float(setRef_dist[GetRandomSurroundCell]) # 이동거리 취득
        # 이동된 셀의 좌표 저장
        MoveResult = np.vstack((MoveResult, Selected_Coord))
        # Set Next Start Coord
        MovedStartCoord = MoveResult[-1]
                
        # 종료기준 계산 
        if MoveResult.tolist().count(MoveResult[-1].tolist()) > 1:
            # print("중복좌표", Selected_Coord, MoveResult.tolist().count(MoveResult[-1].tolist()))
            continue
        else:
            Con_Feeding = round(Con_Feeding - Get_FeedingProp, 5)
        if MoveResult[-1].tolist() == MoveResult[-2].tolist():
            continue
        else:
            Con_Distance = Con_Distance - Get_MovedDist    
        # print("먹이 만족도", Con_Feeding)
        # print("최대이동거리", Con_Distance)
        
        EachPath = MoveResult
    # print("먹이 만족도", Con_Feeding)
    # print("최대이동거리", Con_Distance)
    if (Con_Feeding <= 0) | (Con_Distance <= 0):
        if Con_Feeding <= 0:
            print("먹이를 다 먹었다")
        elif Con_Distance <= 0:
            print("최대이동거리를 다 돌았다")
        else:
            print("Error")
    return EachPath


def SaveAsRaster(InputArray, OutputPath, InputRasterName):
    print("SaveAsRaster, Not Yet")
    


#%%
# 이동 난이도를 예시로 함
ConstInt = 10
Sample_CostMap = np.linspace(1, 10, ConstInt**2).reshape(ConstInt, ConstInt)
Sample_PropMap = np.linspace(0.01, 0.2, ConstInt**2).reshape(ConstInt, ConstInt)

MovedStartCoord = np.array([2, 3])

Testrun = FeedingPropPath(MovedStartCoord, Sample_CostMap, Sample_PropMap, MaxDistance = 1500, cell_size = 50)
print(Testrun)

BaseSize = Sample_CostMap.shape
ZerosArr = np.zeros(BaseSize)


for i in range(0, Testrun.shape[0]):
    ZerosArr[Testrun[i][0]][Testrun[i][1]] = 1



#%%

baseRaster = r"D:\70_PyCode\Wildboar\SampleDATA\results\MinmumDistOneDay.tif"
readRaster = RasterGDAL(baseRaster)

readRaster.write_geotiff( ZerosArr, "D:/70_PyCode/Wildboar/SampleDATA/results/Path.tif")
