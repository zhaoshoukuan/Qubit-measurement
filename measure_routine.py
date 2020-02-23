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




class measure():
    def __init__(self,ats,dc,psg,awg):
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg

    async def S21(self,qubit):
        freq, delta = np.linspace(-4,4,81) + qubit.f_lo, qubit.delta
        await self.psg['psg_lo'].setValue('Output','ON')
        for i in freq:
            await self.psg['psg_lo'].setValue('Frequency', i)
            ch_A, ch_B = await self.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            s = Am + Bm
            yield i-delta, s
        


