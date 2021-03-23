import numpy as np
import math
import matplotlib.pyplot as plt

def Layer(trainData,thresh,setLen):
    DistMat,orderdict = CallDistMat(trainData,setLen)
    # orderdict表示各个类相距最近的类以及距离，key为
    dy_cluid = [n for n in range(setLen)]
    dy_Dot2Clu = {}         #表示各点到类的距离，第一个key为样本点id，第二个key为类id
    for k in range(setLen):
        dy_Dot2Clu[k] = {}
        for m in range(setLen):
            dy_Dot2Clu[k][m] = DistMat[k,m]

    dy_clusdict = {}            
    for n in range(setLen):
        dy_clusdict[n] = [n]            #表示所有类的id
    
    ClusterNum = setLen
    while ClusterNum > thresh:
        mindist,minid = SearchMinDist(orderdict)
        for key,value in orderdict.items():
            if (value[0] == minid[1])and(key != minid[0]):
                orderdict[key] = [minid[0],orderdict[key][1]]
        orderdict.pop(minid[1])
        dy_clusdict[minid[0]] += dy_clusdict[minid[1]]
            
        for id in range(setLen):
            if dy_Dot2Clu[id][minid[1]] == float('inf'):
                dy_Dot2Clu[id][minid[0]] = float('inf')
                dy_Dot2Clu[id].pop(minid[1])
            elif dy_Dot2Clu[id][minid[0]] == float('inf'):
                dy_Dot2Clu[id].pop(minid[1])
            else:
                dy_Dot2Clu[id][minid[0]] = min(dy_Dot2Clu[id][minid[0]],dy_Dot2Clu[id][minid[1]])
                dy_Dot2Clu[id].pop(minid[1])
        new_mindist = float('inf')
        new_minid = -1
        dy_cluid.remove(minid[1])
        for dot in dy_clusdict[minid[0]]:
            for cluid in dy_cluid:
                if new_mindist > dy_Dot2Clu[dot][cluid]:
                    new_mindist = dy_Dot2Clu[dot][cluid]
                    new_minid = cluid
        orderdict[minid[0]] = [new_minid,new_mindist]
        ClusterNum -= 1
        dy_clusdict.pop(minid[1])
    return dy_clusdict

def CallDistMat(trainData,setLen):
    DistMat = float('inf')*np.ones([setLen,setLen])
    orderdict = {}
    for m in range(setLen):
        temp = [m+n for n in range(setLen-m)]
        for n in temp[1:]:
            DistMat[m,n] = CalDist(trainData[...,m],trainData[...,n])
            DistMat[n,m] = DistMat[m,n]
    for m in range(setLen):
        minid = np.argmin(DistMat[...,m])
        orderdict[m] = [minid,DistMat[minid,m]]
    return DistMat,orderdict

def SearchMinDist(orderdict):
    minid = []
    mindist = float('inf')
    for key,value in orderdict.items():
        if value[1] < mindist:
            mindist = value[1]
            minid = [key,value[0]]
    return mindist,minid

def CalDist(vec1,vec2,type = 1):
    dist = 0.0
    if type == 1:
        dist = math.sqrt(np.sum((vec1-vec2)*(vec1-vec2)))
    else:
        dist = 0.0
    return dist


def KMeans(trainData,thresh,setLen,dim):
    RndPerm = np.random.permutation(range(setLen))
    newCenter_1 = trainData[...,RndPerm[0:thresh]]
    cnt = 0
    while True:
        CluDict = UpdateClu(trainData,newCenter_1,setLen) 
        newCenter_2 = UpdateCenter(trainData,CluDict,thresh,dim)
        if ((newCenter_1 == newCenter_2).all()):
            break
        newCenter_1 = newCenter_2
        cnt += 1
    return CluDict

def UpdateClu(trainData,initCenter,setLen):
    newCluDict = {}
    K = np.size(initCenter,1)
    for k in range(K):
        newCluDict[k] = []
    for m in range(setLen):
        mindist = float('inf')
        minid = -1
        for n in range(K):
            dist = CalDist(trainData[...,m],initCenter[...,n])
            if dist < mindist:
                mindist = dist
                minid = n
        newCluDict[minid] += [m] 
    return newCluDict

def UpdateCenter(trainData,CluDict,K,dim):
    newCenter = np.zeros([dim,K])
    for m in range(K):
        temp = np.zeros(dim)
        for n in CluDict[m]:
            temp += trainData[...,n]
        newCenter[...,m] = temp/len(CluDict[m])
    return newCenter
'''
inf = float('inf')
DistMat = np.zeros([5,5])
DistMat[0,...] = np.array([inf,7,2,9,3])
DistMat[1,...] = np.array([7,inf,5,4,6])
DistMat[2,...] = np.array([2,5,inf,8,1])
DistMat[3,...] = np.array([9,4,8,inf,5])
DistMat[4,...] = np.array([3,6,1,5,inf])
orderdict = {0:[2,2],1:[3,4],2:[4,1],3:[1,4],4:[2,1]}
dy_clusdict = Layer(DistMat,orderdict,2,5)
'''

trainData = np.zeros([2,30])
for k in range(30):
    if k < 10:
        trainData[...,k] = np.array([1,2])
    elif k < 20:
        trainData[...,k] = np.array([10,20])
    else:
        trainData[...,k] = np.array([20,0])
    trainData[0,k] += np.random.normal(0,1,1)
    trainData[1,k] += np.random.normal(0,1,1)
CluDict = KMeans(trainData,3,30,2)
print(CluDict)


