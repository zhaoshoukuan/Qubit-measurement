#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   measure_routine.py
@Time    :   2020/02/20 10:24:45
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

import numpy as np, time, pickle
from qulab.wavepoint import WAVE_FORM as WF
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from qulab import waveform_new as wn
from tqdm import tqdm_notebook as tqdm
from qulab.optimize import Collect_Waveform 
from qulab import imatrix as mx


class AWG():
    def __init__(self):
        pass




class qubitCollections():
    def __init__(self,qubits,calimatrix=None):
        self.qubits = {i.q_name:i for i in qubits}
        qstate = {}
        for i in self.qubits:
            qstate[i] = {'dc':0,'ex':0,'read':False}
        self.__setattr__('qstate',qstate)
        if calimatrix != None:
            self.calimatrix = calimatrix
    
    def readMixing(self,f_cavity):
        f_lo = f_cavity.max() + 50e6
        delta =  f_lo - f_cavity 
        n = len(f_cavity)
        return f_lo, delta, n

    def exMixing(self,f_ex):
        ex_lo = f_ex.mean() + 50e6
        delta_ex =  ex_lo - f_ex
        # n = len(f_ex)
        return ex_lo, delta_ex

    def qubitExecute(self,state=None):
        if state != None:
            for i in state:
                self.qstate[i] = state[i]
        else:
            pass
        #bias
        if hasattr(self,'calimatrix'):
            bias = []
            for i,j in enumerate(self.qubits):
                bias.append(state[i]['dc'])
            bias_cross = np.mat(self.calimatrix).I * np.mat(bias).T
            self.bias_cross = bias_cross
        else:
            bias = []
            for i,j in enumerate(self.qubits):
                bias.append(state[i]['dc'])
            self.bias_cross = np.array(bias)
        #read
        fread = []
        for i in state:
            if state[i]['read']:
                fread.append(self.qubits[i].f_lo[0])
        f_lo, delta, n = self.readMixing(np.array(fread))
        self.f_lo, self.delta, self.n = f_lo, delta, n
        #excite
        fex1, fex2 = [], []
        for i in self.qubits:
            if self.qubits[i].inst['ex_lo'] == 'psg_ex1':
                if state[i]['ex'] == 1:
                    fex1.append(self.qubits[i].f_ex[0])
            if self.qubits[i].inst['ex_lo'] == 'psg_ex2':
                if state[i]['ex'] == 1:
                    fex2.append(self.qubits[i].f_ex[0])
        ex_lo1, delta_ex1 = self.exMixing(np.array(fex1))
        ex_lo2, delta_ex2 = self.exMixing(np.array(fex2))
            


