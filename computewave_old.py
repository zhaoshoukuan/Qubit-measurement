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
# 收集变量
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
# 创建波形及sequence
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
# 生成Sequence
################################################################################

async def genSeq(measure,awg,kind):
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind)

################################################################################
# 生成Sequence
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
# awgDC波形
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
# 更新波形
################################################################################

async def writeWave(awg,name,pulse):
    wav_I, wav_Q, mrk1 = pulse
    I, Q, mrk = wav_I(t_new), wav_Q(t_new), mrk1(t_new)
    # I, Q = pulse
    await awg.stop()
    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q, name = name[1])
    await awg.update_marker(name=name[1],mk1=mrk)
    await awg.write('*WAI')

################################################################################
# 波包选择
################################################################################

def whichEnvelope(envelop):
    x = {'square':wn.square,'cospulse':wn.cosPulse,'gaussian':wn.gaussian}
    return x[envelop]

################################################################################
# Rabi波形
################################################################################

async def rabiWave(envelopename='square',nwave=1,amp=1,during=75e-9,shift=200e-9,Delta_lo=80e6,phase=0,phaseDiff=0,DRAGScaling=None):
    
    envelope = whichEnvelope(envelopename)
    wave = (((envelope(during) << (shift+during/2))) + ((envelope(during) << (shift+during/2*3)))) * amp
    # wave = ((wn.cosPulse(2*during) << (shift+during)))
    
    mwav = (wn.square(2*during) << (during+shift)) * amp
    pulse = mpulse = wn.zero()
    for i in range(nwave):
        pulse += wave * amp << 2*i*during
        mpulse += mwav * amp << 2*i*during
    wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)

    return wav_I, wav_Q, mpulse
    # lo2 = Expi(2*np.pi*Delta_lo)
    # # init1 = (((Step(2e-9)<<2*during) - Step(2e-9))<<shift)*lo2
    # init1 = ((CosPulse(2*during)<<during) <<shift)*lo2
    # wav = init1 
    # wav.set_range(*t_range)
    # points = wav.generateData(sample_rate)
    
    # I, Q = np.real(points), np.imag(points)
    # return I, Q

async def Rabi_sequence(measure,kind,awg,t_rabi,nwave=1,envelopename='cospulse'):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    measure.envelopename = envelopename
    await awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc='Rabi_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rabiWave(envelopename=envelopename,nwave=nwave,amp=1,during=i/1e9,phase=0,phaseDiff=0,DRAGScaling=None)
        await writeWave(awg,name_ch,pulse)

################################################################################
# pipulseDetune波形
################################################################################

async def pipulseDetunewave(measure,awg,pilen,axisshift,alpha,name_ch):
    wav_I = wav_Q = mrk1 = wn.zero()
    for i in axisshift:
        pulse = await rabiWave(envelopename=measure.envelopename,during=pilen/1e9,shift=(200+i[0])/1e9,phase=i[1],DRAGScaling=alpha)
        wav_I, wav_Q, mrk1 = wav_I + pulse[0], wav_Q + pulse[1], mrk1 + pulse[2]
    pulse = (wav_I,wav_Q,mrk1)
    await writeWave(awg,name_ch,pulse)

################################################################################
# T1波形
################################################################################

async def T1_sequence(measure,kind,awg,t_rabi,pi_len):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc='T1_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rabiWave(envelopename=measure.envelopename,during=pi_len/1e9,shift=(i+200)*1e-9,phaseDiff=0,DRAGScaling=None)
        await writeWave(awg,name_ch,pulse)

################################################################################
# Ramsey及SpinEcho,CPMG, PDD波形
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
# Ramsey波形
################################################################################

async def ramseyWave(delay,halfpi=75e-9,fdetune=3e6, shift=200e-9, Delta_lo=80e6,envelopename='square'):
    
    envelope = whichEnvelope(envelopename)
    cosPulse1 = ((envelope(halfpi) << (shift+halfpi/2)))
    wav_I1, wav_Q1 = wn.mixing(cosPulse1,phase=2*np.pi*fdetune*delay,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    
    cosPulse2 = ((envelope(halfpi) << (delay+shift+halfpi/2*3)))
    wav_I2, wav_Q2 = wn.mixing(cosPulse2,phase=0.0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = wav_I1 + wav_I2, wav_Q1 + wav_Q2
    
    init1 = wn.square(halfpi) << (shift+halfpi/2)
    init2 = wn.square(halfpi) << (shift+halfpi*3/2+delay)
    mrk = init1 + init2

    return wav_I, wav_Q, mrk
    # lo1 = Expi(2*np.pi*Delta_lo,3*np.pi*1e6*delay)
    # lo2 = Expi(2*np.pi*Delta_lo)
    # # init1 = (((Step(2e-9)<<halfpi) - Step(2e-9))<<shift)*lo1
    # # init2 = (((Step(2e-9)<<halfpi) - Step(2e-9)) << (delay+shift+halfpi))*lo2
    # init1 = ((CosPulse(halfpi)<<halfpi/2)<<shift)*lo1
    # init2 = ((CosPulse(halfpi)<<halfpi/2)  << (delay+shift+halfpi/2*3))*lo2
    # wav = init1 + init2 
    # wav.set_range(*t_range)
    # points = wav.generateData(sample_rate)
    
    # I, Q = np.real(points), np.imag(points)

    # return I, Q

async def Ramsey_sequence(measure,kind,awg,t_rabi,halfpi):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc='Ramsey_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await ramseyWave(delay=i/1e9, halfpi=halfpi/1e9,envelopename=measure.envelopename)
        await writeWave(awg,name_ch,pulse)

################################################################################
# Z_cross波形
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
# Speccrosstalk波形
################################################################################

async def Speccrosstalk_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    await awg_ex.stop()
    await awg_z.stop()
    for j,i in enumerate(tqdm(v_rabi,desc='Speccrosstalk_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        await rabiWave(awg_ex,during=halfpi/1e9,shift=(100+200)*1e-9, name = name_ch)
        await z_crossWave(awg_z,volt=i,during=1100/1e9,shift=(200)/1e9,name=[measure.wave[kind_z][0][j]])
    await awg_ex.write('*WAI')
    await awg_z.write('*WAI')

################################################################################
# AC-Stark波形
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
# ZPulse波形
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
# RB波形
################################################################################

def genXY(during,shift=0e-9,tdelay=0e-9,pulse='pi',envelopename='square'):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    if pulse == 'halfpi':
        pulse = envelope(during) << (shift+during/2)
    if pulse == 'pi':
        pulse = (envelope(during) << (shift+during/2)) + (envelope(during) << (shift+during/2*3))
    
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

async def rbWave(measure,m,gate,pilen,Delta_lo=80e6,shift=0):
    op = {'1':['I'],'2':['X'],'3':['Xhalf'],'4':['Xnhalf'],'5':['Y'],'6':['Yhalf'],'7':['Ynhalf'],
        '8':['X','Y'],'9':['Xhalf','Yhalf','Xnhalf'],'10':['Xhalf','Ynhalf','Xnhalf'],'11':['Ynhalf','X'],
        '12':['Yhalf','X'],'13':['Xhalf','Y'],'14':['Xnhalf','Y'],'15':['Xhalf','Yhalf','Xhalf'],'16':['Xnhalf','Yhalf','Xnhalf'],
        '17':['Xhalf','Yhalf'],'18':['Xnhalf','Yhalf'],'19':['Xhalf','Ynhalf'],'20':['Xnhalf','Ynhalf'],
        '21':['Ynhalf','Xnhalf'],'22':['Ynhalf','Xhalf'],'23':['Yhalf','Xnhalf'],'24':['Yhalf','Xhalf']}

    mseq = mx.cliffordGroup_single(m,gate)
    if mseq == []:
        return
    rotseq = []
    for i in mseq[::-1]:
        rotseq += op[i]
    waveseq_I = waveseq_Q = wav = wn.zero()
    # rotseq = ['Xhalf','Xnhalf','Yhalf','Ynhalf']*m
    # if rotseq == []:
    #     return
    print(rotseq)
    for i in rotseq:
        paras = genParas(i)
        if i == 'I':
            # waveseq_I += wn.zero()
            # waveseq_Q += wn.zero()
            continue
        else:
            pulse = genXY(during=pilen,pulse=paras[1],envelopename=measure.envelopename)
            cosPulse = pulse << shift
            phi = paras[0]
            wav_I, wav_Q = wn.mixing(cosPulse,phase=phi,freq=80e6,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
            waveseq_I += wav_I
            waveseq_Q += wav_Q
            
        if paras[1] == 'pi':
            shift += 2*pilen
        if paras[1] == 'halfpi':
            shift += pilen

    return waveseq_I, waveseq_Q, wn.zero()
    
async def rb_sequence(measure,awg,kind,m,gate,pilen):
    await awg.stop()
    for j,i in enumerate(tqdm(range(len(measure.wave[kind][0])),desc='RB_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rbWave(measure,m,gate,pilen*1e-9)
        await writeWave(awg,name_ch,pulse)
    await awg.query('*OPC?')

################################################################################
# 单比特tomo波形
################################################################################

async def tomoWave(envelopename='square',during=0,shift=200e-9,Delta_lo=80e6,axis='X',DRAGScaling=None):

    envelope = whichEnvelope(envelopename)
    if axis == 'X':
        phi = 0
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=DRAGScaling)
    if axis == 'Y':
        phi = -np.pi/2
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=DRAGScaling)
    if axis == 'Z':
        wav_I = wav_Q = wn.zero()
    return wav_I, wav_Q, wn.zero()

################################################################################
# 单比特tomo测试
################################################################################

async def tomoTest(measure,awg,t,halfpi,axis,name):
    gatepulse = await rabiWave(measure.envelopename,during=t/1e9,shift=(200+halfpi)/1e9,Delta_lo=80e6)
    # gatepulse = await rabiWave(envelopename=measure.envelopename,during=halfpi/1e9)
    tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,Delta_lo=80e6,axis=axis)
    pulse = gatepulse[0]+tomopulse[0], gatepulse[1]+tomopulse[1], wn.zero()
    await writeWave(awg,name,pulse)

################################################################################
# AllXY drag detune
################################################################################

async def dragDetunewave(measure,awg,pilen,coef,axis,name_ch):
    pulse1 = await rabiWave(measure.envelopename,during=pilen/1e9,shift=(200+pilen)/1e9,phase=axis[0],Delta_lo=80e6,DRAGScaling=coef)
    pulse2 = await tomoWave(measure.envelopename,pilen/1e9,Delta_lo=80e6,axis=axis[1],DRAGScaling=coef)
    pulse = pulse1[0] + pulse2[0], pulse1[1] + pulse2[1], pulse1[2] + pulse2[2]
    await writeWave(awg,name_ch,pulse)
    
