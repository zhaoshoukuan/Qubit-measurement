#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ATS.py
@Time    :   2019/04/11 14:52:45
@Author  :   skzhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, zhaogroup-lala
@Desc    :   None
'''


from call.AlazarTechDigitizer import (AlazarTechDigitizer, AlazarTechError, AutoDMA,configure)
import numpy as np                                

samplerate = 1e9
samplesPerRecord = 1024
delta = 50e6

dig = AlazarTechDigitizer(1, 1)
configure(
    dig,
    ARange=1.0,  # V
    BRange=1.0,  # V
    trigLevel=0.6,  # V
    triggerDelay=0,  # Samples
    triggerTimeout=0  # s
    )
def acquiredata():

    with AutoDMA(
            dig,
            samplesPerRecord,
            repeats=5000,
            buffers=None,
            recordsPerBuffer=1,
            timeout=10) as h:
    # do st before aquire data
        chA_list, chB_list = [], []
        for chA, chB in h.read():
        # do st with data
            chA_list.append(chA)
            chB_list.append(chB)
    
    t = np.arange(samplesPerRecord)/samplerate
    y = np.exp(-2j*np.pi*delta*t)
    ch_A = np.array(chA_list).dot(y)
    ch_B = np.array(chB_list).dot(y) 

    
        
    return ch_A, ch_B*1j

def CloseInst(*instrument):  
    for i in instrument:
        if i is not None:
            i.close()
            del i
        
