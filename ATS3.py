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

import zmq,visa,pickle,time
import numpy as np
from call.AlazarTech_Digitizer import *

ctx = zmq.Context.instance()
port = 19873

rm = visa.ResourceManager()

def acquiredata(n,repeats = 1024):
    samplesPerRecord = 1024
    sampleRate = 1e9
    delta = 50e6

    t = np.arange(samplesPerRecord)/sampleRate
    y = np.exp(-2j*np.pi*delta*t)

    awg = rm.open_resource('TCPIP::10.122.7.132')

    dig = AlazarTechDigitizer(1, 1)
    configure(dig, ARange=1.0, BRange=1.0, trigLevel=0.6, triggerDelay=60)

    with DMABufferArray(1, samplesPerRecord*2, repeats) as buffers:   
        
        data1, data2 = [], []
        for i in range(n):
  
            try:

                with AutoDMA2(dig, buffers, samplesPerRecord, repeats, 1, timeout=100) as h:
                    awg.write('TRIG ATR')
                    
                    try:
                        chA_list, chB_list = [], []
                        for chA, chB in h.read():
                            chA_list.append(chA)
                            chB_list.append(chB)
                            
                    finally:
                        pass
            except AlazarTechError as err:
                print(err)
                if err.code == 518:
                    raise SystemExit(1)
    
            ch_A = np.array(chA_list).dot(y)
            ch_B = np.array(chB_list).dot(y)
            data1.append(ch_A.T)
            data2.append(ch_B.T*1j)
            #time.sleep(0.1)
       
    return data1,data2

def CloseInst(*instrument):  
    for i in instrument:
        if i is not None:
            i.close()
            del i
            
if __name__ == '__main__':
    try:
        data = acquiredata(100,501)
        print(data[0][0][0])
        print(len(data[0]))
        print(len(data[0][0]))
    except KeyboardInterrupt:
        raise SystemExit(0)
