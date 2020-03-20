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
            qstate[i] = {'dc':0,'bias_z':0,'ex':0,'read':False}
        self.__setattr__('qstate',qstate)
        if calimatrix != None:
            self.calimatrix = calimatrix
    
    def readMixing(self,f_cavity):
        f_lo = f_cavity.max() + 50e6
        delta =  f_lo - f_cavity 
        n = len(f_cavity)
        return f_lo, delta, n

    def exMixing(self,f):
        qname = [i[0] for i in f]
        f_ex = np.array([i[1] for i in f])
        ex_lo = f_ex.mean() + 50e6
        delta =  ex_lo - f_ex
        delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
        # n = len(f_ex)
        return ex_lo, delta_ex

    def qubitExecute(self,state=None):
#         if state != None:
#             for i in state:
#                 self.qstate[i] = state[i]
#         else:
#             pass
        #bias
        qall = {'dc':{},'awg131':{},'awg132':{},'awg133':{},'awg134':{},'awg_read':{}}
        bias, bias_z = [], []
        for i in self.qubits:
            if i in state and 'dc' in state[i]:
                bias.append(state[i]['dc'])
            else:
                bias.append(self.qstate[i]['dc'])
            if i in state and 'bias_z' in state[i]:
                bias_z.append(state[i]['bias_z'])
            else:
                bias_z.append(self.qstate[i]['bias_z'])
        if hasattr(self,'calimatrix'):
            bias_cross = np.mat(self.calimatrix).I * np.mat(bias).T
            self.bias_cross = bias_cross
            bias_cross_z = np.mat(self.calimatrix).I * np.mat(bias_z).T
            self.bias_cross_z = bias_cross_z
        else:
            self.bias_cross = np.array(bias)
            self.bias_cross_z = np.array(bias_z)
        #read
        fread = []
        for i in state:
            if 'read' in state[i] and state[i]['read']:
                fread.append(self.qubits[i].f_lo[0])
        f_lo, delta, n = self.readMixing(np.array(fread))
        self.f_lo, self.delta, self.n = f_lo, delta, n
        #excite
        fex1, fex2 = [], []
        for i in self.qubits:
            if i in state and 'ex' in state[i]:
                if self.qubits[i].inst['ex_lo'] == 'psg_ex1':
                    if state[i]['ex'] != 0:
                        fex1.append((i,self.qubits[i].f_ex[0]))
                if self.qubits[i].inst['ex_lo'] == 'psg_ex2':
                    if state[i]['ex'] != 0:
                        fex2.append((i,self.qubits[i].f_ex[0]))        

        qall['psg_lo'] = {'Frequncy':self.f_lo}
        qall['awg_read']['ch'+str(7)] = {'Frequency':self.delta}
        qall['awg_read']['ch'+str(8)] = {'Frequency':self.delta}

        for i, j in enumerate(self.qubits):
            qall['dc'][self.qubits[j].inst['dc']] = self.bias_cross[i]
            qall[self.qubits[j].inst['z_awg']]['ch'+str(self.qubits[j].inst['z_ch'])] = {'Volt':self.bias_cross_z[i]}
            if fex1 != []:
                ex_lo1, delta_ex1 = self.exMixing(fex1)
                qall['psg_ex1'] = {'Frequency':ex_lo1}
                if j in delta_ex1:
                    delta_ex = delta_ex1
                    for k in self.qubits[j].inst['ex_ch']:
                        qall[self.qubits[j].inst['ex_awg']]['ch'+str(k)] = {'Frequency':delta_ex[j]}
            if fex2 != []:
                ex_lo2, delta_ex2 = fex2[0][1], {fex2[0][0]:0}
                qall['psg_ex2'] = {'Frequency':ex_lo2}
                if j in delta_ex2:
                    delta_ex = delta_ex2
                    for k in self.qubits[j].inst['ex_ch']:
                        qall[self.qubits[j].inst['ex_awg']]['ch'+str(k)] = {'Frequency':delta_ex[j]}
        self.qall = qall

            


