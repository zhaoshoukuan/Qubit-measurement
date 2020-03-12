#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   imatrix.py
@Time    :   2020/03/12 15:48:18
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import numpy as np 

I = np.mat([[1,0],[0,1]])
X = np.mat([[0,-1j],[-1j,0]])
Xhalf = np.mat([[1,-1j],[-1j,1]]) / np.sqrt(2)
Xnhalf = np.mat([[1,1j],[1j,1]]) / np.sqrt(2)
Y = np.mat([[0,-1],[1,0]])
Yhalf = np.mat([[1,-1],[1,1]]) / np.sqrt(2)
Ynhalf = np.mat([[1,1],[-1,1]]) / np.sqrt(2)

Clifford = {'1':I,
           '2':X,
           '3':Xhalf,
           '4':Xnhalf,
           '5':Y,
           '6':Yhalf,
           '7':Ynhalf,
           '8':X*Y,
           '9':Xhalf*Yhalf*Xnhalf,
           '10':Xhalf*Ynhalf*Xnhalf,
           '11':Ynhalf*X,
           '12':Yhalf*X,
           '13':Xhalf*Y,
           '14':Xnhalf*Y,
           '15':Xhalf*Yhalf*Xhalf,
           '16':Xnhalf*Yhalf*Xnhalf,
           '17':Xhalf*Yhalf,
           '18':Xnhalf*Yhalf,
           '19':Xhalf*Ynhalf,
           '20':Xnhalf*Ynhalf,
           '21':Ynhalf*Xnhalf,
           '22':Ynhalf*Xhalf,
           '23':Yhalf*Xnhalf,
           '24':Yhalf*Xhalf}

def cliffordGroup_single(m):

    mseq = np.random.randint(low=1,high=24,size=m)
    mseq = [str(mseq[i]) for i in range(len(mseq))]
    invertelement = np.mat([[1,0],[0,1]])
    for i in mseq[::-1]:
        invertelement = np.dot(invertelement,Clifford[i])

    invertelement = invertelement.I
    for i in Clifford:
        index0, index1 = np.argwhere(invertelement!=0)[:,0][0], np.argwhere(invertelement!=0)[:,1][0]
        phase = np.angle(invertelement[index0,index1])
        d = invertelement*np.exp(-1j*phase)
        phase2 = np.angle(Clifford[i][index0,index1])
        dtarget = Clifford[i]*np.exp(-1j*phase2)
        if (np.abs(dtarget-d)<0.01).all():
            invertseq = i

#     for i in Clifford:
#         if (np.abs(np.abs(np.dot(invertelement,Clifford[i]))-np.mat([[1,0],[0,1]]))<1e-5).all():
#             invertseq = i
    mseq.append(invertseq)
    return mseq
