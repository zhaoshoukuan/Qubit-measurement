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

# here put the import lib
import numpy as np, time, pickle
from qulab.wavepoint import WAVE_FORM as WF
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from tqdm import tqdm_notebook as tqdm
from qulab.optimize import Collect_Waveform 


t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
### 收集变量
################################################################################

def saveStatus(fname='D:/status.obj'):
    import sys
    
    status = {}
    
    for k in filter(
        lambda s: s[0] != '_' and not callable(globals()[s]) and
        not isinstance(globals()[s], type(sys)) and not s in ['In', 'Out'],
        globals().keys()):
        try:
            status[k] = pickle.dumps(globals()[k])
        except:
            print(k, type(globals()[k]))
            
    with open(fname, 'wb') as f:
         pickle.dump(status, f)
    
def loadStatus(fname='D:/status.obj'):
    with open(fname, 'rb') as f:
        status = pickle.load(f)
    
    for k, v in status.items():
        globals()[k] = pickle.loads(v)

################################################################################
### 创建波形及sequence
################################################################################

async def create_wavelist(measure,kind,para):
    awg,name,n_wave,length = para
    if kind in measure.wave:
        print('kind has already existed')
        return
    @Collect_Waveform(measure.wave,kind)
    async def create_waveformlist(awg,name,n_wave,length):
        name_list = []
        for i in tqdm(name,desc='create_waveformlist'):
            name_collect = []
            for j in range(1,n_wave+1):
                name_w = ''.join((kind,'_',i,'%d'%j))
                name_collect.append(name_w)
                await measure.awg[awg].create_waveform(name=name_w, length=length, format=None) 
            name_list.append(name_collect)
        return name_list
    return create_waveformlist(*para)

################################################################################
### 生成Sequence
################################################################################

async def genSeq(measure,awg,kind):
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind)

################################################################################
### 生成Sequence
################################################################################

async def readSeq(measure,awg,kind):
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind)
    await measure.awg['awg133'].use_sequence(kind,channels=[7,8])
    await measure.awg['awg133'].query('*OPC?')
    await measure.awg['awg133'].output_on(ch=7)
    await measure.awg['awg133'].output_on(ch=8)
    await measure.awg['awg133'].run()
################################################################################
### Rabi波形
################################################################################

async def rabiWave(awg,during=75e-9, shift=200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    lo = Expi(2*np.pi*Delta_lo)
    m1 = CosPulse(during) << during/2
    m2 = CosPulse(during) << during*3/2
    m = m1 + m2
    wav = (m << shift) * lo
    # lo = Expi(2*np.pi*Delta_lo)
    # init = (Step(2e-9)<<during) - Step(2e-9)
    # wav = (init << shift) * lo
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q, name = name[1])
    #await awg.update_marker(name[1],mk1=points1)

async def Rabi_sequence(measure,kind,awg,t_rabi):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='Rabi_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],during=i/1e9, name = name_ch)
    # await measure.awg[awg].run()

################################################################################
### T1波形
################################################################################

async def T1_sequence(measure,kind,awg,t_rabi,pi_len):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='T1_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],during=pi_len/1e9,shift=(i+200)*1e-9, name = name_ch)
    # await measure.awg[awg].run()