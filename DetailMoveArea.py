'''
Filename: d:\70_PyCode\Wildboar\DetailMoveArea.py
Path: d:\70_PyCode\Wildboar
Created Date: Tuesday, July 11th 2023, 9:42:30 pm
Author: Istel

Copyright (c) 2023 Lab.Synergy
'''
#%% lists of used package 
import os 
import numpy as np
# from RasterArcpy import RasterArcpy 
import math

import sys
sys.path.append(r"D:\70_PyCode\TempPythonCode")
sys.path.append(r"D:\70_PyCode")

from AstarBasic import AStar
#%%
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
    # 거리 1 늘린 만큼의 목적 포인트 설정
    RawCircle = circleKernel(ConRadius)
    ExpandCircle = circleKernel(int(ConRadius + 1))
    ExpandRawCircle = np.zeros(ExpandCircle.shape)
    ExpandRawCircle[1:-1, 1:-1] = RawCircle
    GoalGrid = ExpandCircle - ExpandRawCircle
    # print(f"GoalGrid: \n {GoalGrid}")
    #### 적용을 위한 상대좌표 만들기
    origin_idx = GoalGrid.shape[0] //2 , GoalGrid.shape[1] //2
    ref_idx = np.array(Startidx) - np.array(origin_idx)
    Raw_ref_destIdx = np.nonzero(GoalGrid) + np.array(ref_idx).reshape(2,1)
    # 범위를 넘어가는 경우수정
    ref_destinationIdx = np.where(Raw_ref_destIdx < 0, 0, Raw_ref_destIdx )
    ref_destinationIdx = np.where(ref_destinationIdx >= GoalGrid.shape[0], GoalGrid.shape[0], ref_destinationIdx )
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
            print(newRoutes)
        ListConRoutes.append(TupleRouteAndDistance)
    # print(f"ListConRoutes: {ListConRoutes}")
    inner_coordinates = [coordinates for coordinates, _ in ListConRoutes]
    # print(f"{len(inner_coordinates)}, inner_coordinates: {inner_coordinates}")
    Fin_DestIdxs = []
    for Fin_DestIdx in inner_coordinates:
        Fin_DestIdxs.append(Fin_DestIdx[-1])
    # print(len(Fin_DestIdxs))
    return inner_coordinates, Fin_DestIdxs

#%%
def Test_Main():
    grid2 = np.diag(np.ones(30)) 
    grid2[4,4] = 0 
    grid2

    start_P = (8,3)
    Test = GetRoutesConList(start_P, 7, grid2) 

if __name__ == "__main__":
    Test_Main()
