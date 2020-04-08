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
from qulab import waveform_new as wn
from tqdm import tqdm_notebook as tqdm
from qulab.optimize import Collect_Waveform 
from qulab import imatrix as mx


t_list = np.linspace(0,100000,250000)
t_new = np.linspace(-90000,10000,250000)*1e-9
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
    await measure.awg['awgread'].use_sequence(kind,channels=[7,8])
    await measure.awg['awgread'].write('*WAI')
    await measure.awg['awgread'].output_on(ch=7)
    await measure.awg['awgread'].output_on(ch=8)
    await measure.awg['awgread'].run()
    await measure.awg['awgread'].query('*OPC?')

################################################################################
### Rabi波形
################################################################################
async def awgDC(awg,wavname,volt):
    await awg.stop()
    # init = DC(offset=-0.7, length=100e-6, range=(0,1))*0 + volt
    # wav = init
    init  = ((Step(2e-9)<<3000e-9) - Step(2e-9))*volt
    wav = (init >> 2250e-9)
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    await awg.update_waveform(points,wavname)

################################################################################
### Rabi波形
################################################################################

async def rabiWave(awg,amp=1, during=75e-9,nwave=1, shift=200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    # lo = Expi(2*np.pi*Delta_lo)
    # m1 = CosPulse(during) << during/2
    # m2 = CosPulse(during) << during*3/2
    # m = m1 + m2
    # wav = (m << shift) * lo
    
    # lo = Expi(2*np.pi*Delta_lo)
    # init = (Step(2e-9)<<2*during) - Step(2e-9)
    # wav = (init << shift) * lo

    # wav.set_range(*t_range)
    # points = wav.generateData(sample_rate)
    # I, Q = np.real(points), np.imag(points)

    # pulse = wn.square(2*during) << (during+shift)
    # wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.14312,DRAGScaling=None)
    pulse = wn.zero()
    for i in range(nwave):
        pulse += (((wn.cosPulse(during) << (shift+during/2))) + ((wn.cosPulse(during) << (shift+during/2*3)))) * amp << 2*i*during
    wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.14312,DRAGScaling=1/135e6/2)
    I, Q = wav_I(t_new), wav_Q(t_new)

    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q*0.9423, name = name[1])

    init1 = (Step(0e-9) << 2*during) - Step(0e-9)
    wav1 = (init1 << shift) 
    wav1.set_range(*t_range)
    points1 = wav1.generateData(sample_rate)
    await awg.update_marker(name=name[1],mk1=points1)
    # init2 = (Step(0e-9) << 2.1*during) - Step(0e-9)
    # wav2 = (init2 << -30e-9) 
    # wav2.set_range(*t_range)
    # points2 = wav2.generateData(sample_rate)
    # await awg.update_marker(name='Readout_I',mk1=points2)

async def Rabi_sequence(measure,kind,awg,t_rabi,nwave=1):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    await measure.awg[awg].query('*OPC?')
    for j,i in enumerate(tqdm(t_rabi,desc='Rabi_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],amp=1,during=i/1e9,nwave=nwave, name = name_ch)
    await measure.awg[awg].query('*OPC?')

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
    await measure.awg[awg].write('*WAI')

################################################################################
### Ramsey及SpinEcho,CPMG, PDD波形
################################################################################

async def coherenceWave(awg,t_run,during=75e-9,n_wave=0,seqtype='CPMG',detune=3e6,shift=200e-9,Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    pulse1 = wn.square(during) << (during/2+shift)
    wavI1, wavQ1 = wn.mixing(pulse1,phase=2*np.pi*detune*t_run,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    pulse3 = wn.square(during) << (t_run+(2*n_wave+1.5)*during+shift)
    wavI3, wavQ3 = wn.mixing(pulse3,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'CPMG':
        pulse2, step = wn.zero(), t_run / n_wave
        for i in range(n_wave):
            pulse2 += wn.square(2*during) << ((i+0.5)*step+(i+1)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'PDD':
        pulse2, step = wn.zero(), t_run / (n_wave + 1)
        for i in range(n_wave):
            pulse2 += wn.square(2*during) << ((i+1)*step+(i+1)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wavI, wavQ = wavI1 + wavI2 + wavI3, wavQ1 + wavQ2 + wavQ3
    I, Q = wavI(t_new), wavQ(t_new)
    await awg.update_waveform(I, name[0])
    await awg.update_waveform(Q, name[1])

async def Coherence_sequence(measure,kind,awg,t_rabi,halfpi,n_wave=0,seqtype='CPMG',detune=3e6):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='Ramsey_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await coherenceWave(measure.awg[awg],i/1e9,halfpi/1e9,n_wave,seqtype,detune,name=name_ch)
    await measure.awg[awg].write('*WAI')

################################################################################
### Ramsey波形
################################################################################

async def ramseyWave(awg,delay,halfpi=75e-9,fdetune=0e6, shift=200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    # lo2, lo1 = Expi(2*np.pi*Delta_lo), Expi(2*np.pi*Delta_lo,2*np.pi*fdetune*delay)
    # m1 = (CosPulse(halfpi) << halfpi/2)
    # m2 = (CosPulse(halfpi) << (halfpi*3/2 + delay))  #m1,m2,lo1,lo2所乘位置要考察
    # wav = (m1 + m2) << shift

    # lo2, lo1 = Expi(2*np.pi*Delta_lo), Expi(2*np.pi*Delta_lo,2*np.pi*fdetune*delay)
    # m1 = ((Step(2e-9)<<halfpi) - Step(2e-9) << shift) *lo1
    # m2 = (((Step(2e-9)<<halfpi) - Step(2e-9)) << (shift+halfpi + delay)) * lo2
    # wav = (m1 + m2) 

    # wav.set_range(*t_range)
    # points = wav.generateData(sample_rate)
    # I, Q = np.real(points), np.imag(points)
    
    cosPulse1 = ((wn.cosPulse(halfpi) << (shift+halfpi/2)))
    wav_I1, wav_Q1 = wn.mixing(cosPulse1,phase=2*np.pi*fdetune*delay,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    cosPulse2 = ((wn.cosPulse(halfpi) << (delay+shift+halfpi/2*3)))
    wav_I2, wav_Q2 = wn.mixing(cosPulse2,phase=0.0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = wav_I1 + wav_I2, wav_Q1 + wav_Q2
    I, Q = wav_I(t_new), wav_Q(t_new)

    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q, name = name[1])
    #await awg.update_marker(name[1],mk1=points1)

    init1 = (Step(0e-9)<<halfpi) - Step(0e-9) 
    init2 = ((Step(0e-9)<<halfpi) - Step(0e-9)) << (delay+halfpi)
    wav1 = (init1 + init2) << shift
    wav1.set_range(*t_range)
    points1 = wav1.generateData(sample_rate)
    await awg.update_marker(name=name[1],mk1=points1)

async def Ramsey_sequence(measure,kind,awg,t_rabi,halfpi):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='Ramsey_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await ramseyWave(measure.awg[awg], delay=i/1e9, halfpi=halfpi/1e9, name = name_ch)
    await measure.awg[awg].write('*WAI')

################################################################################
### Z_cross波形
################################################################################

async def z_crossWave(awg,volt,during=200e-9, shift=200e-9,name=['q1_z']):

    init = ((Step(2e-9)<<during) - Step(2e-9)) * volt
    wav = init << shift 
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)

    await awg.update_waveform(points, name[0])
    
async def Z_cross_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg_ex].stop()
    await measure.awg[awg_z].stop()
    for j,i in enumerate(tqdm(v_rabi,desc='Z_cross_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        await ramseyWave(measure.awg[awg_ex], delay=300/1e9, halfpi=halfpi/1e9, fdetune=0, name = name_ch)
        await z_crossWave(measure.awg[awg_z],volt=i,during=200/1e9,shift=(50+halfpi+200)/1e9,name=[measure.wave[kind_z][0][j]])
    await measure.awg[awg_ex].write('*WAI')
    await measure.awg[awg_z].write('*WAI')

################################################################################
### Speccrosstalk波形
################################################################################

async def Speccrosstalk_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg_ex].stop()
    await measure.awg[awg_z].stop()
    for j,i in enumerate(tqdm(v_rabi,desc='Speccrosstalk_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        await rabiWave(measure.awg[awg_ex],during=halfpi/1e9,shift=(100+200)*1e-9, name = name_ch)
        await z_crossWave(measure.awg[awg_z],volt=i,during=1100/1e9,shift=(200)/1e9,name=[measure.wave[kind_z][0][j]])
    await measure.awg[awg_ex].write('*WAI')
    await measure.awg[awg_z].write('*WAI')

################################################################################
### AC-Stark波形
################################################################################

async def ac_stark_sequence(measure,awg,kind,pilen,t_rabi):
    awg_read, awg_ex = measure.awg['awgread'], measure.awg[awg]
    #await awg_ex.stop()
    await awg_read.stop() 
    lo = Expi(2*np.pi*50e6)
    m1 = (Step(0e-9)<<measure.wavelen*1e-9) - Step(0e-9) >> measure.wavelen*1e-9
    m2 = ((Step(0e-9)<<500*1e-9) - Step(0e-9)) << 3000e-9
    wav = (m1 + m2) * lo
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg_read.update_waveform(I, name = 'Readout_I')
    await awg_read.update_waveform(Q, name = 'Readout_Q')
    await awg_read.use_waveform(name='Readout_I',ch=7)
    await awg_read.use_waveform(name='Readout_Q',ch=8)
    await awg_read.write('*WAI')
    await awg_read.output_on(ch=7)
    await awg_read.output_on(ch=8)
    await awg_read.run()

    # for j,i in enumerate(tqdm(t_rabi,desc='acStark_sequence')):
    #     name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
    #     await rabiWave(awg_ex,during=pilen/1e9,shift=i*1e-9, name = name_ch)
    # await awg_ex.query('*OPC?')
    
################################################################################
### ZPulse波形
################################################################################

async def zWave(awg,name,ch,volt,during=500e-9,shift=1000e-9):

    await awg.stop()
    wav = (((Step(0e-9) << during) - Step(0e-9)) << shift) * volt
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    await awg.update_waveform(points,name=name[0])
    await awg.use_waveform(name=name[0],ch=ch)
    await awg.output_on(ch=ch)
    await awg.write('WAI')
    await awg.run()
    await awg.query('*OPC?')

################################################################################
### RB波形
################################################################################

# def genXY(phi,during,pulse,shift=0e-9,tdelay=0e-9,Delta_lo=80e6):
#     shift += 300e-9
#     if pulse == 'pi':
#         # m1 = CosPulse(during) << during/2
#         # m2 = CosPulse(during) << (during*3/2+tdelay)
#         # m = m1 + m2
#         #lo = Expi(2*np.pi*Delta_lo)
#         m1 = (Step(2e-9)<<during) - Step(2e-9)
#         m2 = ((Step(2e-9)<<during) - Step(2e-9)) << during
#         m = (m1 + m2)
#     if pulse == 'halfpi':
#         # m = CosPulse(during) << during/2
#         m = (Step(2e-9)<<during) - Step(2e-9)
#     wav = (m << shift) 
        
#     return wav

def genXY(phi,during,shift=0e-9,tdelay=0e-9,pulse='pi'):
    shift += 200e-9
    if pulse == 'halfpi':
        pulse = wn.cosPulse(during) << (shift+during/2)
    if pulse == 'pi':
        pulse = (wn.cosPulse(during) << (shift+during/2)) + (wn.cosPulse(during) << (shift+during/2*3))
    
    return pulse

def genParas(x):
    if x == 'I':
        paras = (0,'pi')
    elif x == 'X':
        paras = (0,'pi')
    elif x == 'Xhalf':
        paras = (0,'halfpi')
    elif x == 'Xnhalf':
        paras = (np.pi,'halfpi')
    elif x == 'Y':
        paras = (np.pi/2,'pi')
    elif x == 'Yhalf':
        paras = (np.pi/2,'halfpi')
    elif x == 'Ynhalf':
        paras = (3*np.pi/2,'halfpi')
    return paras

async def rbWave(awg,m,pilen,Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    op = {'1':['I'],'2':['X'],'3':['Xhalf'],'4':['Xnhalf'],'5':['Y'],'6':['Yhalf'],'7':['Ynhalf'],
        '8':['X','Y'],'9':['Xhalf','Yhalf','Xnhalf'],'10':['Xhalf','Ynhalf','Xnhalf'],'11':['Ynhalf','X'],
        '12':['Yhalf','X'],'13':['Xhalf','Y'],'14':['Xnhalf','Y'],'15':['Xhalf','Yhalf','Xhalf'],'16':['Xnhalf','Yhalf','Xnhalf'],
        '17':['Xhalf','Yhalf'],'18':['Xnhalf','Yhalf'],'19':['Xhalf','Ynhalf'],'20':['Xnhalf','Ynhalf'],
        '21':['Ynhalf','Xnhalf'],'22':['Ynhalf','Xhalf'],'23':['Yhalf','Xnhalf'],'24':['Yhalf','Xhalf']}

    mseq = mx.cliffordGroup_single(m)
    if mseq == []:
        return
    rotseq = []
    for i in mseq[::-1]:
        rotseq += op[i]
    shift, waveseq_I, waveseq_Q, wav = 0, 0, 0, 0
    # rotseq = ['Xhalf']*m
    # if rotseq == []:
    #     return
    # print(rotseq)
    for i in rotseq:
        paras = genParas(i)
        if i == 'I':
            waveseq_I += wn.zero()
            waveseq_Q += wn.zero()
        else:
            pulse = genXY(phi=paras[0],during=pilen,pulse=paras[1])
            cosPulse = pulse << shift
            phi = paras[0]
            wav_I, wav_Q = wn.mixing(cosPulse,phase=0.0,freq=80e6,ratioIQ=1.0,phaseDiff=0.0,DRAGScaling=None)
            waveseq_I += wav_I
            waveseq_Q += wav_Q
            # wav_I, wav_Q = genXY(phi=paras[0],during=pilen,pulse=paras[1])
            # phi = paras[0]
            # waveseq_I += (wav_I << shift) * wn.cos(2*np.pi*Delta_lo,phi)
            # waveseq_Q += (wav_Q << shift) * wn.sin(2*np.pi*Delta_lo,phi)
            # w = genXY(phi=paras[0],during=pilen,pulse=paras[1])
            # phi = paras[0]
            # lo = Expi(2*np.pi*80e6,phi)
            # wav += (w << shift) * lo
        if paras[1] == 'pi':
            shift += 2*pilen
        if paras[1] == 'halfpi':
            shift += pilen
    I, Q = waveseq_I(t_new), waveseq_Q(t_new)
    # wav.set_range(*t_range)
    # points = wav.generateData(sample_rate)
    # I, Q = np.real(points), np.imag(points)
    await awg.update_waveform(I, name[0])
    await awg.update_waveform(Q, name[1])
    
async def rb_sequence(measure,awg,kind,m,pilen):
    awg = measure.awg[awg]
    await awg.stop()
    await awg.query('*OPC?')
    for j,i in enumerate(tqdm(range(len(measure.wave[kind][0])),desc='RB_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rbWave(awg,m,pilen*1e-9,name=name_ch)
    await awg.query('*OPC?')

################################################################################
### 单比特tomo波形
################################################################################

def tomoWave(during,shift=200e-9,Delta_lo=80e6,axis='X'):
    
    if axis == 'X':
        phi = 0
    if axis == 'Y':
        phi = -np.pi/2
    cosPulse = wn.cosPulse(during) << (shift+during/2)
    wav_I, wav_Q = wn.mixing(cosPulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    
    return wav_I, wav_Q

################################################################################
### 单比特tomo测试
################################################################################

async def tomoTest(awg,halfpi,axis,name,shift=400e-9):
    cosPulse = wn.cosPulse(halfpi) << (shift+halfpi/2)
    wav_I, wav_Q = wn.mixing(cosPulse,phase=0,freq=80e6,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if axis == 'Z':
        I, Q = wav_I, wav_Q
    if axis == 'X':
        I, Q = wav_I + tomoWave(halfpi,axis='X')[0], wav_Q + tomoWave(halfpi,axis='X')[1]
    if axis == 'Y':
        I, Q = wav_I + tomoWave(halfpi,axis='Y')[0], wav_Q + tomoWave(halfpi,axis='Y')[1]
    
    await awg.update_waveform(I,name[0])
    await awg.update_waveform(Q,name[1])