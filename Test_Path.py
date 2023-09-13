'''
Filename: d:\70_PyCode\Wildboar\Test_Path.py
Path: d:\70_PyCode\Wildboar
Created Date: Tuesday, September 5th 2023, 2:39:47 pm
Author: Istel

Copyright (c) 2023 Lab.Synergy
'''
#%%
import numpy as np
import datetime
import sys
sys.path.append(r"D:\70_PyCode\TempPythonCode")
sys.path.append(r"D:\70_PyCode")

from RasterGDAL import RasterGDAL


#%% 완료된 함수정의
def ToSelectProp(SurroundProp):
    resistances = SurroundProp
    # 각 저항의 저항값 계산 (역수)
    inverse_resistances = 1 / resistances
    # 전류 비율 계산
    total_inverse_resistance = np.sum(inverse_resistances)
    current_ratios = inverse_resistances / total_inverse_resistance
    return current_ratios

# 경로 하나 만들기
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

        #### 종료기준 계산 
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
    # if (Con_Feeding <= 0) | (Con_Distance <= 0):
    #     if Con_Feeding <= 0:
    #         print("먹이를 다 먹었다")
    #     elif Con_Distance <= 0:
    #         print("최대이동거리를 다 돌았다")
    #     else:
    #         print("Error")
    return EachPath


#### 경로 1000개 도출
def GetPathRatio(GetStartCoord, CostMAP, FeedingMAP, MaxDistance = 1500, cell_size = 30, MaximumPahts = 1000):
    BaseSize = CostMAP.shape
    MaxPahts = MaximumPahts
    FullPaths = np.zeros((MaxPahts, BaseSize[0], BaseSize[1]))
    CountPahts = 0
    while CountPahts < MaxPahts:
        EachPath = FeedingPropPath(GetStartCoord, CostMAP, FeedingMAP, MaxDistance, cell_size)
        EachPathRaster = np.zeros(BaseSize)
        for i in range(0, EachPath.shape[0]):
            EachPathRaster[EachPath[i][0]][EachPath[i][1]] = 1
        FullPaths[CountPahts] = EachPathRaster
        CountPahts += 1
        # 진행상황 10% 단위로 출력
        # if CountPahts % (MaxPahts/10) == 0:
            # print(f"전체 {MaxPahts}개의 결과 중 현재{CountPahts}개의 경로가 생성되었습니다.")
    # PropotionPath
    GetCount = np.sum(FullPaths, axis=0)
    GetRatio = np.sum(FullPaths, axis=0) / MaxPahts
    return GetRatio, GetCount


def SaveAsRaster(InputArray, OutputPath, InputRasterName):
    print("SaveAsRaster, Not Yet")



#%%
#
BasePath = r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data\jaeyeon_Extent"
os.chdir(BasePath)

Startraster = r"StartPoint.tif"
FeedingRaster = r".\results\ProportionOfStay.tif"
CostMap = r"CostMap_Temp.tif"

readRaster = RasterGDAL(Startraster)
# CannotMove = np.where(readRaster.RasterToArray()[0] == 1, 9999, 0)
#
ReadRaster_Feeding = RasterGDAL(FeedingRaster)
FeedingArr = ReadRaster_Feeding.RasterToArray()[0]
#
readCost_Imp = RasterGDAL(CostMap)
# TempArr = readCost_Imp.RasterToArray()[0] * -1
# ReverseTempArr = TempArr + 1
# TempCost = np.where(readRaster.RasterToArray()[0] == 1, 9999, ReverseTempArr)
TempCost = readCost_Imp.RasterToArray()[0]


#%%
# TestFunc
# Startraster = RasterGDAL( r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data\Sample_Extent\Boundary_Extent.tif")
StartCoords = readRaster.RasterToArray()[0] 
GetStartP = np.argwhere(StartCoords == 1) 
GetStartCoord = GetStartP[1000] 

Sub_GetStartP = GetStartP[0:100] 

# 
BaseStack = np.zeros((len(GetStartP), TempCost.shape[0], TempCost.shape[1]))
BaseStack_Sum = np.zeros((len(GetStartP), TempCost.shape[0], TempCost.shape[1]))
for i in range(0, len(GetStartP)):
    GetStartCoord = GetStartP[i]
    TempArray = GetPathRatio(GetStartCoord, TempCost, FeedingArr, MaxDistance = 1500, cell_size = 10, MaximumPahts = 1000)
    RatioArray = TempArray[0]
    # BaseStack[i] = np.where(RatioArray == 0, 1, RatioArray)
    BaseStack[i] = RatioArray
    BaseStack_Sum[i] = TempArray[1]
    # 진행상황의 1% 단위로 출력
    if i % (len(GetStartP)/10) == 0:
        # 현재 시간을 얻어옵니다.
        current_time = datetime.datetime.now() 
        # 시간을 원하는 형식으로 출력합니다.
        formatted_time = current_time.strftime("%H:%M:%S") 
        print(f"전체 {len(GetStartP)}개의 결과 중 현재{i}번째 경로가 생성되었습니다.", "현재 시간:", formatted_time)

# 빈도수의 합으로 정리 
ToAllRatio = np.sum(BaseStack, axis=0)
SumToAllRatio = np.sum(BaseStack_Sum, axis=0)

#%%
# save np.array to csv
np.savetxt(r".\results\AllRatio.csv", ToAllRatio, delimiter=",")
np.savetxt(r".\results\SumToAllRatio.csv", SumToAllRatio, delimiter=",")


 
# 합으로 빈도수 확인 
ToAllRatio = np.sum(NonZeros, axis=0)
ToFin = np.where(ToAllRatio == len(GetStartP), 0, ToAllRatio)

# Upscale = ToAllRatio * 1000000
ToFin = np.where(ToAllRatio == 1, 0, ToAllRatio)
Normalizing = ToFin / np.max(ToFin)


ZeroToNull = np.where(ToFin == 0, None, ToFin)

# Save As Tiff
readCost_Imp.write_geotiff(ToFin, r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data\Sample_Extent\results\ToFin.tif")

readCost_Imp.write_geotiff(ToAllRatio, r".\results\TestFinRun20230913.tif")

readCost_Imp.write_geotiff(SumToAllRatio, r".\results\TestFinRun20230913_Sum.tif")

# Testrun = GetPathRatio(GetStartCoord, TempCost, FeedingArr, MaxDistance = 1500, cell_size = 30, MaximumPahts = 10)
# print(Testrun)
# print(np.unique(Testrun, return_counts=True))





#%%
# 이동 난이도를 예시로 함
ConstInt = 10
Sample_CostMap = np.linspace(1, 10, ConstInt**2).reshape(ConstInt, ConstInt)
Sample_PropMap = np.linspace(0.01, 0.2, ConstInt**2).reshape(ConstInt, ConstInt)

MovedStartCoord = np.array([2, 3])

BaseSize = Sample_CostMap.shape
ZerosArr = np.zeros(BaseSize)





#%% 1000개의 경로 도출 


MaximumPahts = 5

PathRaster = np.zeros(BaseSize)
FullPaths = np.zeros((MaximumPahts, BaseSize[0], BaseSize[1]))
CountPahts = 0
while CountPahts < MaximumPahts:
    EachPath = FeedingPropPath(GetStartCoord, TempCost, FeedingArr, MaxDistance = 1500, cell_size = 30)
    EachPathRaster = np.zeros(BaseSize)
    for i in range(0, EachPath.shape[0]):
        EachPathRaster[EachPath[i][0]][EachPath[i][1]] = 1
    FullPaths[CountPahts] = EachPathRaster
    CountPahts += 1
    print(f"전체 {MaximumPahts}개의 결과 중 현재{CountPahts}개의 경로가 생성되었습니다.")

# PropotionPath
GetRatio = np.sum(FullPaths, axis=0) / MaximumPahts

Testrun = FeedingPropPath(GetStartCoord, TempCost, FeedingArr, MaxDistance = 1500, cell_size = 30)
print(len(Testrun))  


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

