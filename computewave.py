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
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from tqdm import tqdm_notebook as tqdm


t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9


################################################################################
### Rabi波形
################################################################################

async def rabiWave(awg,during=75e-9, shift=-200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    def genXY(during, phi, shift, Delta_lo=80e6):
        lo = Expi(2*np.pi*Delta_lo, phi)
        m1 = CosPulse(during) << during/2
        m2 = CosPulse(during) << during*3/2
        m = m1 + m2
        wav = (m >> shift) * lo
        wav.set_range(*t_range)
        points = wav.generateData(sample_rate)
        return np.real(points), np.imag(points)
    
    I, Q = genXY(during,0,shift,Delta_lo)
    await awg.update_waveform(I, name[0])
    await awg.update_waveform(Q, name[1])
    #await awg.update_marker(name[1],mk1=points1)

async def Rabi_sequence(measure,kind,awg,t_rabi=None):
    if t_rabi == None:
        t_rabi = np.linspace(0,100,501)
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in tqdm(enumerate(t_rabi,start=0),desc='Rabi_sequence'):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],during=i/1e9, name = name_ch)