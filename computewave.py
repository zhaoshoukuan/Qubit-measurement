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
import numpy as np, time
from qulab.wavepoint import WAVE_FORM as WF
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian
from tqdm import tqdm_notebook as tqdm


t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
### Rabi波形
################################################################################

async def rabiWave(awg,during=75e-9, shift=-200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
        lo = Expi(2*np.pi*Delta_lo)
        #init = CosPulseDrag(during,alpha=0.5,Delta=266e6) << 0.5*during
        init = (Step(2e-9)<<during) - Step(2e-9)
        wav = (init >> shift) * lo
        wav.set_range(*t_range)
        points = wav.generateData(sample_rate)
        init1 = (Step(0e-9)<<during) - Step(0e-9)
        wav1 = (init1 >> shift) 
        wav1.set_range(*t_range)
        points1 = wav1.generateData(sample_rate)
    
        I, Q = np.real(points), np.imag(points)

        await awg.update_waveform(I, name[0])
        await awg.update_waveform(Q, name[1])
        await awg.update_marker(name[1],mk1=points1)

async def updateWaveform_rabi(measure,kind,awg,t_rabi=None):
    if t_rabi == None:
        t_rabi = np.linspace(0,100,501)
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in tqdm(enumerate(t_rabi,start=0),desc='Rabi_sequence'):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await updateWaveform_rabi(measure.awg[awg],during=i/1e9, name = name_ch)