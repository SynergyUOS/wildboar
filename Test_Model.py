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
import math
from collections import defaultdict

import sys
sys.path.append(r"D:\70_PyCode\TempPythonCode")
sys.path.append(r"D:\70_PyCode")

from Wildboar.core import runCore
from RasterGDAL import RasterGDAL
from AstarBasic import AStar


#%% Sample Run 
dirpath = r"D:\70_PyCode\Wildboar\SampleDATA"
os.chdir(dirpath)
#
InDEM = r"Match_KoreaChina_1arcDEM2.tif"
InputConProp = r"Jinju_KOFTR31_PA1_Full_MAXENT.tif"
TestResult = runCore(dirpath, InDEM, InputConProp, 42000, SearchDistance = 1000 )

#%% 
dirpath = r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data"
os.chdir(dirpath)
#### Input 
#
InDEM = r"북한산DEM.tif"
InputConProp = r"TempProp.tif"
TestResult = runCore(dirpath, InDEM, InputConProp, 42000, SearchDistance = 1000 )


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
InputConProp = r"ModelINPUT\Jinju_KOFTR31_PA1_Full_MAXENT.tif"
TestResult = runCore(dirpath, InDEM, InputConProp, 42000, SearchDistance = 1000 )

#%%
dirpath = r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data\jaeyeon_Extent"
os.chdir(dirpath)

#### Input 
## Input_DEM 데이터 파일
InDEM = r"DEM_.tif"
InputConProp = r"TempProportion.tif"

TestResult = runCore(dirpath, InDEM, InputConProp, 42000, SearchDistance = 1000 )


#%% Define Function
#### Get Goalindex / GetModifiedGoalIndex
direction_vector_to_distance = {
    (-1, 0): 1,  # 방향과 길이임 (-1, 0) 방향으로 가면서 이경우 길이는 1이라는 의미임
    (1, 0): 1,
    (0, -1): 1,
    (0, 1): 1,
    (-1, -1): math.sqrt(2), # comment out when needed 
    (-1, 1): math.sqrt(2), # comment out when needed
    (1, -1): math.sqrt(2), # comment out when needed
    (1, 1): math.sqrt(2), # comment out when needed
}

def circleKernel(x):
    radius = x
    size = 2 * radius + 1
    radiusSquarred= radius**2
    mask = np.zeros((size, size))
    distance = lambda x, y: (x-radius)**2 + (y-radius)**2
    for i in range(size):
        for j in range (size):
            if distance(i, j) <= radiusSquarred:
                mask[i, j] = 1
    return mask


# 특정셀에서 지나갈 수 없는 부분을 반영하여 만족하는 거리만큼의 목적지 좌표를 찾는 함수
# 주의: 이동이 불가능한 지역에서 시작할 수 없음
def GetRoutesConList(Startidx, ConRadius, ObstacleGrid):
    StartP_arr = np.array(Startidx)
    # 거리 1 늘린 만큼의 목적 포인트 설정
    RawCircle = circleKernel(ConRadius)
    ExpandCircle = circleKernel(int(ConRadius + 1))
    ExpandRawCircle = np.zeros(ExpandCircle.shape)
    ExpandRawCircle[1:-1, 1:-1] = RawCircle
    GoalGrid = ExpandCircle - ExpandRawCircle
    # print(ExpandCircle.shape)
    #### 적용을 위한 상대좌표 만들기
    origin_idx = ExpandCircle.shape[0]//2, ExpandCircle.shape[1]//2
    ref_idx = np.array(Startidx) - np.array(origin_idx)
    Raw_ref_destIdx = np.nonzero(GoalGrid) + np.array(ref_idx).reshape(2,1)
    # print(f"Raw_ref_destIdx: {Raw_ref_destIdx}")
    # 범위를 넘어가는 경우수정
    ref_destinationIdx = np.where(Raw_ref_destIdx < 0, 0, Raw_ref_destIdx )
    ref_destinationIdx = np.where(ref_destinationIdx >= ObstacleGrid.shape[0], ObstacleGrid.shape[0], ref_destinationIdx )
    ####
    GoalPointidx = np.transpose(ref_destinationIdx)
    GoalIndexList = [] # 모든 목적지 좌표(튜플) 리스트
    # print(f"GoalPointidx: {GoalPointidx}")
    #### 장애물 그리드에 목적지가 있는 경우 제외
    ObstacleGrididx = np.transpose(np.nonzero(ObstacleGrid))
    for i in GoalPointidx.tolist():
        # print(i, type(i))
        check_str = str(i)
        TargetListStr = [str(x) for x in ObstacleGrididx.tolist()]
        if check_str in TargetListStr:
            pass
        else:
            GoalIndexList.append(i)
    # print(f"GoalIndexList: {GoalIndexList}")
    #### 각 목적지 까지의 경로 찾기
    ListRoutes_NeedReduce = [] 
    ListConRoutes = [] 
    for i in GoalIndexList:
        GetRoutes = [Startidx] 
        ReadAstar = AStar(ObstacleGrid) 
        Goalindex = tuple(i)
        # print(f"Goalindex: {Goalindex}")
        routes, distance = ReadAstar.search(Startidx, Goalindex)
        GetRoutes += routes 
        GetRoutes.append(Goalindex) 
        # print(f"EachRoute: {GetRoutes}, {distance}") # distance * cellSize 하면 길이 나옴
        if distance > ConRadius :  # 기준 반지름 반경을 초과하면
            ListRoutes_NeedReduce.append((GetRoutes, distance))
        else:
            ListConRoutes.append((GetRoutes, distance))
    # print(f"ListConRoutes: {ListConRoutes}")
    # print(f"ListRoutes_NeedReduce: {ListRoutes_NeedReduce}")

    #### 거리줄이기 필요한 경로 기준 경로만큼 거리 줄이기
    for OverRadius in ListRoutes_NeedReduce:
        Criteria = ConRadius
        PathDistance = OverRadius[1]
        newRoutes = OverRadius[0]
        while max(Criteria, PathDistance) == PathDistance:
            # 마지막과 마지막 바로 전 좌표의 차이를 통해 위치관계를 파악
            Refindex = np.array(newRoutes[-1]) - np.array(newRoutes[-2])
            getDistance = direction_vector_to_distance[tuple(Refindex)]
            PathDistance = PathDistance - getDistance
            newRoutes = newRoutes[:-1]
            TupleRouteAndDistance = (newRoutes, PathDistance)
            # print(newRoutes)
        ListConRoutes.append(TupleRouteAndDistance)
    # print(f"ListConRoutes: {ListConRoutes}")
    inner_coordinates = [coordinates for coordinates, _ in ListConRoutes]
    # print(f"{len(inner_coordinates)}, inner_coordinates: {inner_coordinates}")
    Fin_DestIdxs = []
    for Fin_DestIdx in inner_coordinates:
        Fin_DestIdxs.append( Fin_DestIdx[-1] )
    delTuple_Fin_DestIdxs = [[item for item in sublist] for sublist in Fin_DestIdxs]
    # print(len(Fin_DestIdxs))
    return inner_coordinates, delTuple_Fin_DestIdxs


#%%
dirpath = r"E:\Dropbox\03.Research_Projects\2023_첨단기술을 이용한 멧돼지 서식지 분석 및 모니터링\99_Data\Sample_Extent"
os.chdir(dirpath)
#### Input 
 
InDEM = r"DEM_Extent.tif"
InputConProp = r"TempPropModify_Extent.tif"
TestResult = runCore(dirpath, InDEM, InputConProp, 42000, SearchDistance = 1000 )


#%% Sample 
def GetResultsProp(Startidx, ConRadius, ObstacleGrid, InputConProp):
    StartP_arr = np.array(Startidx)
    # 거리 1 늘린 만큼의 목적 포인트 설정
    RawCircle = circleKernel(ConRadius)
    CircleToindex = np.argwhere(RawCircle == 1)
    # print(f"CircleToindex: {CircleToindex}")
    CircleCenter = np.array((RawCircle.shape[0]//2, RawCircle.shape[1]//2))
    ExpandCircleCenter = np.tile(CircleCenter, (CircleToindex.shape[0], 1))
    Ref_kernelIdx = CircleToindex - ExpandCircleCenter   
    ExpandStartP = np.tile(Startidx, (CircleToindex.shape[0], 1))
    Ref_TrueKernel = ExpandStartP + Ref_kernelIdx
    CLipCoord = np.clip(Ref_TrueKernel, [0, 0], [ObstacleGrid.shape[0], ObstacleGrid.shape[0]])
    ToList_ClipCoord = CLipCoord.tolist()
    # print(f"ToList_ClipCoord: {ToList_ClipCoord}")
    #### 장애물 그리드에 목적지가 있는 경우 제외
    ObstacleGrididx = np.argwhere(ObstacleGrid == 1)
    for i in ObstacleGrididx:
        for coord in CLipCoord:
            if np.all(coord == i):
                # print(f"Remove{coord.tolist()}")
                ToList_ClipCoord.remove(coord.tolist())
    ClipCoord = np.array(ToList_ClipCoord) 
    unique_CLipCoord = np.unique(ClipCoord, axis=0)          
    # print(f"CLipCoord: {len(CLipCoord)}")
    # print(f"unique_CLipCoord: {len(unique_CLipCoord)}")
    # print(f"unique_CLipCoord: {unique_CLipCoord}")
    sumProp = 0
    for coord in unique_CLipCoord:
        getProp = InputConProp[coord[0], coord[1]]
        # print(f"getProp: {getProp}")
        sumProp += getProp
    print(f"sumProp: {sumProp}")
    return unique_CLipCoord



CostMap = r"FillBuilding_Extent.tif"
# BoundaryMap = r"Reclass_공원경계.tif"
MinmumRadius = r"Radious_Extent.tif"
ResultProp = r".\results\ProportionOfStay.tif"

Read_CostMap = RasterGDAL(CostMap).RasterToArray()[0]
Read_MinmumRadius = RasterGDAL(MinmumRadius).RasterToArray()[0]
Read_TempProp = RasterGDAL(InputConProp).RasterToArray()[0]
Read_ResultProp = RasterGDAL(ResultProp).RasterToArray()[0]


indexarg = np.argwhere(Read_MinmumRadius != 0)
indexarg[1000]

Startidx = tuple(indexarg[1000])
getRadius = int(Read_MinmumRadius[Startidx[0], Startidx[1]])


# print(Startidx, getRadius)
Paths = GetResultsProp(tuple(Startidx), getRadius-5, Read_CostMap, Read_ResultProp)


emptyArr = np.zeros(Read_ResultProp.shape)
for i in Paths:
    emptyArr[i[0], i[1]] = 1

SaveToimg = RasterGDAL(ResultProp).write_geotiff(emptyArr, r".\results\TestResult2.tif")


#%%
# 좌표 어레이의 차이
a = np.array([1,1])
b = np.array([3,-5])
C = b - a
# C의 차원알고 싶음





#%%
CostMap = r"Fill_Building.tif"
BoundaryMap = r"Reclass_공원경계.tif"
MinmumRadius = r".\results\fill_minmum.tif"

Read_CostMap = RasterGDAL(CostMap).RasterToArray()[0]
Read_BoundaryMap= RasterGDAL(BoundaryMap).RasterToArray()[0]
Read_MinmumRadius = RasterGDAL(MinmumRadius).RasterToArray()[0]

getIndex = np.argwhere(Read_BoundaryMap == 1)
getIndex

# start_P = tuple(getIndex[50])
# start_P

# Test = GetRoutesConList(start_P, 5, Read_CostMap)

allCells = []
runCount = 0
for Startidx in getIndex:
    runCount += 1
    if len(getIndex) % runCount == 0:
        print(f"Processing: {round(runCount/len(getIndex)*100, 2)}%")
    getRadius = int(Read_MinmumRadius[Startidx[0], Startidx[1]])
    # Error 무시
    try:
        Paths = GetRoutesConList(tuple(Startidx), getRadius, Read_CostMap)
    except:
        pass    
    allCells += Paths[1]

len(allCells)

SaveToCSV = pd.DataFrame(allCells)
SaveToCSV.to_csv(r".\results\AllCells.csv", index=False)
#%%  



#%%
# 방향에 따른 이동거리 딕셔너리

gridMap = np.zeros((7,7))
A = circleKernel(3)
A
sub = circleKernel(2)
b = np.zeros( A.shape )
b[1:-1, 1:-1] = sub
C = A - b
# Get Goalindex
Cindex = np.transpose(np.nonzero(C))

for i in Cindex:
    Startidx = (3,3) # start index
    GetRoutes = [Startidx]
    ReadAstar = astar = AStar(gridMap)
    Goalindex = tuple(i)
    routes, distance = ReadAstar.search(Startidx, Goalindex)
    GetRoutes += routes
    GetRoutes.append(Goalindex)
    print(f"EachRoute: {GetRoutes}")

# 대각선으로 장애물이 있는 경우
grid2 = np.diag(np.ones(7))
grid2idx = np.transpose(np.nonzero(grid2))
grid2idx



#%%
def Test_Main():
    ## Basic Usage: Astar Class를 읽은 뒤, astar = AStar(gridmap)으로 객체를 생성하고, astar.search(start, goal)로 경로를 찾는다.
    # test 1
    gridmap = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    start = (0, 1)
    goal = (4, 3)
    astar = AStar(gridmap) # Read Astar Class
    routes, distance = astar.search(start, goal) # Search Path from start to goal
    print("Routes:", " -> ".join(map(str, [start, *routes, goal])))
    print("Distance:", distance)
    # test 2
    gridmap = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    start = (0, 4)
    goal = (4, 4)
    astar = AStar(gridmap)
    routes, distance = astar.search(start, goal)
    print("Routes:", " -> ".join(map(str, [start, *routes, goal])))
    print("Distance:", distance)
    # test 3
    gridmap = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    start = (3, 2)
    goal = (5, 3)
    astar = AStar(gridmap)
    routes, distance = astar.search(start, goal)
    print("Routes:", " -> ".join(map(str, [start, *routes, goal])))
    print("Distance:", distance)


if __name__ == "__main__":
    Test_Main()


#%% 
def GetRoutesConList(GoalIndexList):
    # 경로 찾기
    List_needCut = []
    ConRoutes = []
    for i in GoalIndexList:
        GetRoutes = [Startidx]
        ReadAstar = AStar(grid2)
        Goalindex = tuple(i)
        routes, distance = ReadAstar.search(Startidx, Goalindex)
        GetRoutes += routes
        GetRoutes.append(Goalindex)
        print(f"EachRoute: {GetRoutes}, {distance}") # distance * cellSize 하면 길이 나옴
        if distance > BaseRadius :  # 기준 반지름 반경을 초과하면
            List_needCut.append((GetRoutes, distance))
        else:
            ConRoutes.append((GetRoutes, distance))
            #
    # 거리 조절하기
    for OverRadius in List_needCut:
        Criteria = BaseRadius
        PathDistance = OverRadius[1]
        newRoutes = OverRadius[0]
        # print(OverRadius)
        # print(np.array(OveaRadius[0][-1]), np.array(OveaRadius[0][-2]))
        while max(Criteria, PathDistance) == PathDistance:
            # print(PathDistance)
            # 마지막과 마지막 바로 전 좌표의 차이를 통해 위치관계를 파악
            Refindex = np.array(newRoutes[-1]) - np.array(newRoutes[-2])
            getDistance = direction_vector_to_distance[tuple(Refindex)]
            # print(getDistance)
            PathDistance = PathDistance - getDistance
            # print(PathDistance)
            newRoutes = newRoutes[:-1]
            TupleRouteAndDistance = (newRoutes, PathDistance)
            print(newRoutes)
        ConRoutes.append(TupleRouteAndDistance)
    inner_coordinates = [coordinates for coordinates, _ in ConRoutes]
    return inner_coordinates

def GetModifiedGoalIndex(RoutesArray):
    Destinations = []
    for Routeidx in range(len(RoutesArray)):
        Startidx, Denstinationidx = RoutesArray[Routeidx][0], RoutesArray[Routeidx][-1]
        Destinations.append(Denstinationidx)
    return Destinations


   
GoalPointidx = np.transpose(np.nonzero(C))
checkIdx_list = []
for i in GoalPointidx.tolist():
    print(i, type(i))
    check_str = str(i)
    TargetListStr = [str(x) for x in grid2idx.tolist()]
    if check_str in TargetListStr:
        pass
    else:
        checkIdx_list.append(i)


# 원하는 거리를 초과하는 경로 도출
BaseRadius = 3
Startidx = (3,4)

List_needCut = []
ConRoutes = []
for i in checkIdx_list:
    GetRoutes = [Startidx]
    ReadAstar = AStar(grid2)
    Goalindex = tuple(i)
    routes, distance = ReadAstar.search(Startidx, Goalindex)
    GetRoutes += routes
    GetRoutes.append(Goalindex)
    print(f"EachRoute: {GetRoutes}, {distance}") # distance * cellSize 하면 길이 나옴
    if distance > BaseRadius :  # 기준 반지름 반경을 초과하면
        List_needCut.append((GetRoutes, distance))
    else:
        ConRoutes.append((GetRoutes, distance))
        #

List_needCut
ConRoutes

# 거리 조절하기
for OverRadius in List_needCut:
    Criteria = BaseRadius
    PathDistance = OverRadius[1]
    newRoutes = OverRadius[0]
    # print(OverRadius)
    # print(np.array(OveaRadius[0][-1]), np.array(OveaRadius[0][-2]))
    while max(Criteria, PathDistance) == PathDistance:
        # print(PathDistance)
        # 마지막과 마지막 바로 전 좌표의 차이를 통해 위치관계를 파악
        Refindex = np.array(newRoutes[-1]) - np.array(newRoutes[-2])
        getDistance = direction_vector_to_distance[tuple(Refindex)]
        # print(getDistance)
        PathDistance = PathDistance - getDistance
        # print(PathDistance)
        newRoutes = newRoutes[:-1]
        TupleRouteAndDistance = (newRoutes, PathDistance)
        print(newRoutes)
    ConRoutes.append(TupleRouteAndDistance)


inner_coordinates = [coordinates for coordinates, _ in ConRoutes]
len(inner_coordinates)

for Route in range(len(inner_coordinates)):
    print(Route, inner_coordinates[Route])

####

RoutesDict = defaultdict(list)


# print( OveaRadius[0] )
# reduceDensity = OveaRadius[0][:-1]
# print( reduceDensity )





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



