import numpy as np
import math
import perceptron as per
def LinearBoost(trainData,lables,thresh):
    setLen = np.size(trainData,1)
    dim = np.size(trainData,0)
    w = 1/setLen*np.ones(setLen)
    error_rate = 1
    total_alpha = []
    total_coe = []
    total_b = []
    cnt = 0
    while error_rate >= thresh:
        y_in = lables*w
        if cnt == 0:
            coe,b = per.Percptron(trainData.T,y_in)
            total_coe += [list(coe)]
            error_rate,temp_lable = CalError(trainData,setLen,lables,1,np.array(total_coe),b)
        else:
            coe,b = per.Percptron(trainData.T,y_in)
            total_coe += [list(coe)]
            error_rate,temp_lable = CalError(trainData,setLen,lables,np.array(total_alpha),np.array(total_coe),total_b)
        alpha = 1/2*math.log((1-error_rate)/(error_rate))
        total_alpha += [alpha]
        total_b += [b]
        Zm = np.dot(w,np.exp(-alpha*lables*temp_lable))
        w = w*np.exp(-alpha*lables*temp_lable)/Zm
        cnt += 1
    return total_alpha,total_coe,total_b

def Sgn(x):
    return (x>0)-(x<0)

def CalError(trainData,setLen,lables,alpha,w,b):
    error_num = 0
    temp_lable = np.array(setLen)
    for k in range(setLen):
        temp_lable[k] = Sgn(np.dot(alpha,np.matmul(w,trainData[...,k])+b))
        if temp_lable[k] != lables[k]:
            error_num += 1
    error_rate = error_num/setLen
    return error_rate,temp_lable

if __name__ == '__main__':
    trainData = np.array([[0,1,2,3,4,5,6,7,8,9]])
    lables = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
    alpha,coe,b = LinearBoost(trainData,lables,0.1)
    print(alpha,coe,b)