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
import numpy as np, serial, time
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF
from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab import computewave as cw

t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

class common():
    def __init__(self,freqall,ats,dc,psg,awg,jpa):
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.wave = {}
        self.att = {}
        self.inststate = 0
        self.t_list = np.linspace(0,100000,250000)
        self.t_range = (-90e-6,10e-6)
        self.sample_rate = 2.5e9
        
class qubitCollections():
    def __init__(self,qubits,q_target=None):
        self.qubits = {i.q_name:i for i in qubits}
        if q_target != None:
            qasdict = self.qubits[q_target]._asdict()
            for i in qasdict:
                # if not hasattr(self,i):
                self.__setattr__(i,qasdict[i])
        else:
            self.inst = {i.q_name:i.inst for i in qubits}
            self.q_name = 'allqubits'

        self.f_lo = np.array([i.f_lo[0] for i in qubits])
        self.f_ex = np.array([i.f_ex[0] for i in qubits])

    def qubitExecute(self,q_target=[]):
        q = []
        for i in q_target:
            q.append(self.qubits[i])
        return q

################################################################################
# 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,measure,com):
        self.com = com
        try:
            ser = serial.Serial(self.com,baudrate=115200, parity='N',bytesize=8, stopbits=1, timeout=1)
            self.ser = ser
        except:
            self.ser = measure.att[com]
        measure.att[com] = self.ser
        if self.ser.isOpen():    # make sure port is open     
            print(self.ser.name + ' open...')
            # self.ser.write(b'*IDN?\n')
            # x = self.ser.readline().decode().split('\r''\n')
            # print(x[0])
            self.ser.write(b'ATT?\n')
            y = self.ser.readline().decode().split('\r''\n')
            print('last ATT',y[0])

    def close(self):
        self.ser.close()

    def Att(self,att,closeinst=True):
        self.ser.write(b'ATT %f\n'%att)
        time.sleep(1)
        self.ser.write(b'ATT?\n')
        y = self.ser.readline().decode().split('\r''\n')
        print('now ATT',y[0])

        if closeinst:
            self.close()
 
################################################################################
# AWG同步
################################################################################

async def awgSync(measure):
    t_list = measure.t_list
    wf = WF(t_list)
    for i in measure.awg:
        await measure.awg[i].create_waveform(name=''.join((i,'_sync1')),length=len(measure.t_list),format=None)
        await measure.awg[i].create_sequence(name=''.join((i,'_syncSeq')),steps=3,tracks=1)
        height,t_end,width = [1],[50000],[2000]
        sample = wf.square_wave(t_end,width,height)
        await measure.awg[i].update_waveform(sample,name=''.join((i,'_sync1')))
        for j in range(3):
            j += 1
            goto = 'FIRST' if j == 3 else 'NEXT'
            await measure.awg[i].set_sequence_step(name=''.join((i,'_syncSeq')),sub_name=[''.join((i,'_sync1'))],\
                                                   step=j,wait='ATR',goto=goto,repeat=1,jump=None)
        await measure.awg[i].use_sequence(name=''.join((i,'_syncSeq')),channels=[1])
        await measure.awg[i].query('*OPC?')
        await measure.awg[i].output_on(ch=1)
        await measure.awg[i].setValue('Force Jump',1,ch=1)
        await measure.awg[i].run()  

################################################################################
# 腔频率设置
################################################################################

async def resn(f_cavity):
    f_lo = f_cavity.max() + 50e6
    delta =  f_lo - f_cavity 
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
# 开关JPA
################################################################################

async def jpa_switch(measure,state='OFF'):
    if state == 'ON':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','ON')
        await measure.psg[measure.jpa.inst['pump']].setValue('Frequency',(measure.jpa.f_ex))
        await measure.psg[measure.jpa.inst['pump']].setValue('Power',measure.jpa.power_ex)
        await measure.dc[measure.jpa.inst['dc']].DC(measure.jpa.bias)
    if state == 'OFF':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','OFF')
        await  measure.dc[measure.jpa.inst['dc']].DC(0)

################################################################################
# 关闭仪器
################################################################################

async def close(measure):
    inst = {**measure.dc , **measure.psg}
    for i in inst:
        await inst[i].setValue('Output','OFF')

################################################################################
# 查询仪器状态
################################################################################

async def QueryInst(measure):
    inst = {**measure.dc , **measure.psg}
    state = {}
    for i in inst:
        try:
            if 'dc' in i:
                current = await inst[i].getValue('Offset')
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n'))  
                load = 'high Z' if load != 50 else 50
                err = (await inst[i].query('syst:err?')).strip('\n').split(',')
                sm = {'offset':current,'load':load,'error':err[0]}
                state[i] = sm
            if 'psg' in i:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                Mform = await inst[i].getValue('Mform')
                err = (await inst[i].query('syst:err?')).strip('\n').split(',')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output,'moutput':Moutput,\
                    'mform':Mform,'error':err[0]}
                state[i] = sm
        finally:
            pass
    measure.inststate = state
    return state

################################################################################
# 初始化仪器
################################################################################

async def InitInst(measure,psgdc=True,awgch=True,clearwaveseq=None):
    if psgdc:
        await close(measure)
    if awgch:
        for i in measure.awg:
            for j in range(8):
                await measure.awg[i].output_off(ch=j+1)
    if clearwaveseq != None:
        for i in clearwaveseq:
            await measure.awg[i].stop()
            #await measure.awg[i].query('*OPC?')
            await measure.awg[i].clear_waveform_list()
            await measure.awg[i].clear_sequence_list()
            m = list(measure.wave.keys())
            for j in m:
                if i in j:
                    del measure.wave[j]
                if i == 'awg134':
                    del measure.wave['Read']


################################################################################
# 恢复仪器最近状态
################################################################################

async def RecoverInst(measure,state=None):
    if state is None:
        state = measure.inststate
    for i in state:
        if 'dc' in i:
            await measure.dc[i].DC(state[i]['offset'])
        if 'psg' in i:
            await measure.psg[i].setValue('Frequency',eval(state[i]['freq'].strip('GHz'))*1e9)
            await measure.psg[i].setValue('Power',eval(state[i]['power'].strip('dBm')))
            if state[i]['output'] == '1':
                await measure.psg[i].setValue('Output','ON')
            else:
                await measure.psg[i].setValue('Output','OFF')

################################################################################
# S21
################################################################################

async def S21(qubit,measure,modulation=False,freq=None):
    #await jpa_switch(measure,state='OFF')
    if freq == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-2.5,2.5,126)*1e6 + f_lo
        if modulation:
            await cw.modulation_read(measure,delta,tdelay=measure.readlen)
    await measure.psg['psg_lo'].setValue('Output','ON')
    await cw.couldRun(measure.awg['awgread'])
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield i-delta, s

async def test(measure,n):
    for i in range(n):
        time.sleep(0.1)
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield i-np.array([50e6]), s

################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,126)*1e6+f_lo
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','OFF')
    job = Job(S21, (qubit,measure,modulation),auto_save=False,max=126)
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cw.ats_setup(measure.ats,delta)
        await cw.modulation_read(measure,delta,tdelay=measure.readlen)
        base = 0
        for i in range(25):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            base += Am + Bm
        base /= 25
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base,f_s21, s_s21

################################################################################
# S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,modulation=False):

    for i in current:
        await measure.dc[qubit.inst['dc']].DC(i)
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    # for i in current:
    #     clist = np.mat(calimatrix).I * np.mat([0,i,0,0]).T
    #     for k, j in enumerate(clist,start=2):
    #         await measure.dc[qubit.inst['q%d'%k]['dc']].DC(j)
    #     job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
    #     f_s21, s_s21 = await job.done()
    #     n = np.shape(s_s21)[1]
    #     yield [i]*n, f_s21, s_s21
################################################################################
# S21vsFlux_awgoffset
################################################################################

async def S21vsFlux_awgoffset(qubit,measure,current,calimatrix,modulation=False):
    # for i in current:
    #     await measure.awg[qubit.inst['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z')), length=len(measure.t_list), format=None)
    #     await measure.awg[qubit.inst['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z')),ch=qubit.inst['z_ch'])
    #     await measure.awg[qubit.inst['z_awg']].query('*OPC?')
    #     await measure.awg[qubit.inst['z_awg']].output_on(ch=qubit.inst['z_ch'])
    #     await measure.awg[qubit.inst['z_awg']].setValue('Offset',i,ch=qubit.inst['z_ch'])
    #     # await cw.awgDC(measure.awg[qubit.inst['z_awg']],''.join((qubit.q_name,'_z')),i)
    #     await measure.awg[qubit.inst['z_awg']].run()
    #     await measure.awg[qubit.inst['z_awg']].query('*OPC?')
    #     job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
    #     f_s21, s_s21 = await job.done()
    #     n = np.shape(s_s21)[1]
    #     yield [i]*n, f_s21, s_s21
    for i in current:
        clist = np.mat(calimatrix).I * np.mat([0,i,0,0]).T
        for k, j in enumerate(clist,start=2):
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z%d'%k)), length=len(measure.t_list), format=None)
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z%d'%k)),ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].output_on(ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].setValue('Offset',j,ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].run()
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att,com='com7',modulation=False):
    att_setup = Att_Setup(com)
    for i in att:
        att_setup.Att(i,closeinst=False)
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    att_setup.close()

################################################################################
# SingleSpec
################################################################################

async def singlespec(qubit,measure,freq,modulation=False,readponit=True):
    if readponit:
        f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation)
    else:
        n, base = len(measure.delta), measure.base
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in freq:
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base

################################################################################
# SingleSpec扫电压
################################################################################

async def specbias(qubit,measure,ftarget,bias,modulation=False):
    await measure.dc[qubit.inst['dc']].DC(round(np.mean(bias),3))
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',ftarget)
    f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation)
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in bias:
        await measure.dc[qubit.inst['dc']].DC(i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base

################################################################################
# Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,init,modulation=False):
    current = np.linspace(-qubit.T_bias[0]*0.3,qubit.T_bias[0]*0.3,31) + qubit.T_bias[1] 
    # current = np.linspace(-1.759*0.3,1.759*0.3,31) + 0.224
    # # current = np.linspace(-0.8,0.8,33)
    # qubit.inst['ex_awg'] = 'awg132'
    # qubit.inst['ex_ch'] = [7, 8]
    # qubit.inst['ex_lo'] = 'psg_ex2'
    await cw.modulation_ex(qubit,measure)
    for i in current:
        await measure.dc[qubit.inst['dc']].DC(i)
        # clist = np.mat(calimatrix).I * np.mat(init).T * i
        # for k, j in enumerate(clist,start=2):
        #     await measure.dc[qubit.inst['q%d'%k]['dc']].DC(j)
        # # clist = np.mat(calimatrix).I * np.mat(init).T * i
        # # for k, j in enumerate(clist,start=2):
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z%d'%k)), length=len(measure.t_list), format=None)
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z%d'%k)),ch=qubit.inst['q%d'%k]['z_ch'])
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].output_on(ch=qubit.inst['q%d'%k]['z_ch'])
        # #     await cw.zWave(measure.awg[qubit.inst['q%d'%k]['z_awg']],[''.join((qubit.q_name,'_z%d'%k))],qubit.inst['q%d'%k]['z_ch'],j[0,0],5000e-9,200e-9)
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].run()
        # #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        job = Job(singlespec, (qubit,measure,freq,modulation),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Rabi
################################################################################

async def rabi(qubit,measure,t_rabi,comwave=False,amp=1,nwave=1):
    name = ''.join((qubit.inst['ex_awg'],'coherence_rabi'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.Rabi_sequence(measure,name,awg,t_rabi,amp,nwave)

    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    await cw.readSeq(measure,measure.awg['awgread'],'Read')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

################################################################################
# Rabi_waveform
################################################################################

async def Rabi_waveform(qubit,measure,t_rabi,nwave=1):

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in t_rabi:
        pulse = await cw.rabiWave(envelopename='cospulse',during=i/1e9,Delta_lo=300e6)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield [i], s-measure.base

################################################################################
# Rabi_population
################################################################################

async def rabiPop(qubit,measure,t_rabi,which):
    # await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in t_rabi:
        await cw.rabiWave(awg,during=i/1e9,name=qname)
        await cw.couldRun(awg)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        ss = s[:,0]
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict(d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# 优化pi脉冲
################################################################################
    
async def pipulseOpt(qubit,measure,nwave,wavlen):
    pilen = qubit.pi_len
    t = np.linspace(0.5*pilen,1.5*pilen,wavlen)
    for i in range(nwave):
        job = Job(rabi, (qubit,measure,t,True,1,2*i+1), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [2*i+1]*measure.n, t_r, s_r

################################################################################
# pi脉冲振幅
################################################################################
    
async def pipulseAmp(qubit,measure,t_rabi):
    amplitude = np.linspace(0.4,1,25)
    for i in amplitude:
        job = Job(rabi, (qubit,measure,t_rabi,True,i,1), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [i]*measure.n, t_r, s_r

################################################################################
# 优化pi脉冲detune
################################################################################
    
async def detuneOpt(qubit,measure,which,alpha,nwave=10):
    pilen, freq = qubit.pi_len, np.linspace(-10,10,101)*1e6 + qubit.delta_ex[0]
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    # await cw.pipulseDetunewave(measure,awg,pilen,[[0,-np.pi],[2*pilen,0]],alpha,qname)
    # await cw.couldRun(awg)
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in freq:
        # await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        pulselist = await cw.pipulseDetunewave(measure,awg,pilen,nwave,alpha,i,qname)
        await cw.couldRun(awg)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0], ch_B[:,0]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict(d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# AllXY drag detune
################################################################################
    
async def AllXYdragdetune(qubit,measure,which,alpha):
    pilen, coef = qubit.pi_len, np.linspace(-2,2,41)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    for j in [[0,'Y'],[-np.pi/2,'X']]:
        for i in coef:
            await cw.dragDetunewave(measure,awg,pilen,i/alpha,j,qname)
            await cw.couldRun(awg)
            # ch_A, ch_B = await measure.ats.getIQ()
            # Am, Bm = ch_A[:,0], ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            # ss = Am + Bm
            # d = list(zip(np.real(ss),np.imag(ss)))
            # y = measure.predict(d)
            # pop = list(y).count(which)/len(y)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            s = Am + Bm
            yield [i], s-measure.base

################################################################################
# 优化IQ-Mixer相位
################################################################################
    
async def IQphaseOpt(qubit,measure,which):
    pilen, angle = qubit.pi_len, np.linspace(-np.pi/2,np.pi/2,201)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    # await awg.create_waveform(name=qname[0], length=len(t_list), format=None)
    # await awg.create_waveform(name=qname[1], length=len(t_list), format=None)

    # await awg.use_waveform(name=qname[0],ch=qubit.inst['ex_ch'][0])
    # await awg.use_waveform(name=qname[1],ch=qubit.inst['ex_ch'][1])
    # await awg.query('*OPC?')
    # await awg.output_on(ch=qubit.inst['ex_ch'][0])
    # await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in angle:
        pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,phaseDiff=i)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0], ch_B[:,0]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict(d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# Rabi_power
################################################################################

async def rabiPower(qubit,measure,att):
    name, awg = [qubit.q_name+'_I', qubit.q_name+'_Q'], measure.awg[qubit.inst['ex_awg']]
    # await awg.create_waveform(name=name[0], length=len(t_list), format=None)
    # await awg.create_waveform(name=name[1], length=len(t_list), format=None)
    await cw.genwaveform(measure,awg,name,qubit.inst['ex_ch'])
    await cw.rabiWave(awg,during=25e-9,name=name)
    # await awg.use_waveform(name=name[0],ch=qubit.inst['ex_ch'][0])
    # await awg.use_waveform(name=name[1],ch=qubit.inst['ex_ch'][1])
    # await awg.output_on(ch=qubit.inst['ex_ch'][0])
    # await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.couldRun(awg)
    # att_setup = Att_Setup(qubit.inst['com'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    n = len(measure.base)
    for i in att:
        # att_setup.Att(i,False)
        await measure.psg[qubit.inst['ex_lo']].setValue('Power',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield [i]*n, s - measure.base
    # att_setup.close()

################################################################################
# IQ-Mixer优化
################################################################################

async def optIQMixer(qubit,measure,att):
    name, awg = [qubit.q_name+'_I', qubit.q_name+'_Q'], measure.awg[qubit.inst['ex_awg']]
    # await awg.create_waveform(name=name[0], length=len(t_list), format=None)
    # await awg.create_waveform(name=name[1], length=len(t_list), format=None)
    await cw.genwaveform(measure,awg,name,qubit.inst['ex_ch'])
    await cw.rabiWave(awg,during=25e-9,name=name)
    # await awg.use_waveform(name=name[0],ch=qubit.inst['ex_ch'][0])
    # await awg.use_waveform(name=name[1],ch=qubit.inst['ex_ch'][1])
    # await awg.output_on(ch=qubit.inst['ex_ch'][0])
    # await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await cw.couldRun(awg)
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    # att_setup = Att_Setup(qubit.inst['com'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    n = len(measure.base)
    for i in att:
        # att_setup.Att(i,False)
        await measure.psg[qubit.inst['ex_lo']].setValue('Power',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield [i]*n, s - measure.base
    # att_setup.close()

################################################################################
# Z_Pulse（Z与Readout时序矫正)
################################################################################

async def zPulse(qubit,measure,t):
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    exname = [qubit.q_name+'_I', qubit.q_name+'_Q']
    freq = np.linspace(-200,200,201)*1e6 + qubit.f_ex[0] + qubit.delta_ex[0]
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    
    await cw.genwaveform(measure,z_awg,zname,zch)
    await cw.zWave(z_awg,zname,zch,0.4)
    await cw.couldRun(z_awg)
    await cw.genwaveform(measure,ex_awg,exname,qubit.inst['ex_ch'])

    n = len(measure.delta)
    for i in t:
        pulse = await cw.rabiWave(ex_awg,qubit.pi_len/1e9,shift=i/1e9,name=exname)
        await cw.writeWave(ex_awg,exname,pulse)
        await cw.couldRun(ex_awg)
        job = Job(singlespec,(qubit,measure,freq,False,False),max=len(freq),auto_save=False)
        f, s = await job.done()
        #n = np.shape(s)[1] 
        yield [i]*n, f-qubit.delta_ex, s

################################################################################
# Z_Pulse（Z与Readout时序矫正) population
################################################################################

async def zPulse_pop(qubit,measure,t):
    awg = measure.awg[qubit.inst['ex_awg']]
    freq = np.linspace(-200,200,201)*1e6 + qubit.f_ex[0] + qubit.delta_ex[0]
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=500)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await measure.awg[qubit.inst['z_awg']].create_waveform(name=zname[0], length=len(t_list), format=None)  
    await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,0.4)
    Iname, Qname = qubit.q_name+'_I', qubit.q_name+'_Q'
    await awg.create_waveform(name=Qname, length=len(t_list), format=None)
    await awg.create_waveform(name=Iname, length=len(t_list), format=None)
    await awg.use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    n = len(measure.delta)
    for i in t:
        await awg.stop()
        await cw.rabiWave(awg,qubit.pi_len/1e9,shift=i/1e9,name=[Iname,Qname])
        await awg.run()
        await awg.query('*OPC?')
        # ch_A, ch_B = await measure.ats.getIQ()
        # s = ch_A + 1j*ch_B
        # s = np.mean(s)
        # #ss, which = s[:,0], 1
        # #d = list(zip(np.real(ss),np.imag(ss)))
        # #y = measure.predict(d)
        # #pop = list(y).count(which)/len(y)
        # yield [i]*n, [s]*n
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield [i]*n, s - measure.base

################################################################################
# AC-Stark（XY与Readout时序矫正)
################################################################################

async def singleacStark(qubit,measure,t_rabi,Delta_lo=80e6):
    name =  ''.join((qubit.inst['ex_awg'],'coherence')) #name--kind
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    
    await cw.ac_stark_wave(measure,measure.awg['awgread'])
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    await cw.T1_sequence(measure,name,awg,t_rabi,qubit.pi_len,0,Delta_lo)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base
    
async def acStark(qubit,measure,t_rabi):
    freq = np.linspace(-30,30,121)*1e6 + qubit.delta_ex
    for i in freq:
        job = Job(singleacStark, (qubit,measure,t_rabi,i), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [i]*measure.n, t_r, s_r

################################################################################
# 优化读出点
################################################################################

async def readOp(qubit,measure,modulation=False,freq=None):
    pilen = qubit.pi_len
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,DRAGScaling=-1.2/272e6/2/np.pi)
    await cw.writeWave(awg,qname,pulse)
    await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,modulation),auto_save=False,max=126)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
# 优化读出长度
################################################################################

async def readWavelen(qubit,measure):
    pilen, t = qubit.pi_len, np.linspace(900,2000,21,dtype=np.int64)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    # await awg.create_waveform(name=qname[0], length=len(t_list), format=None)
    # await awg.create_waveform(name=qname[1], length=len(t_list), format=None)
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9)
    await cw.writeWave(awg,qname,pulse)
    # await awg.use_waveform(name=qname[0],ch=qubit.inst['ex_ch'][0])
    # await awg.use_waveform(name=qname[1],ch=qubit.inst['ex_ch'][1])
    # await awg.query('*OPC?')
    # await awg.output_on(ch=qubit.inst['ex_ch'][0])
    # await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await cw.couldRun(awg)
    for k in t:
        await cw.modulation_read(measure,measure.delta,tdelay=int(k),repeats=5000)
        state = []
        for j, i in enumerate(['OFF','ON']):
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            s = Am + Bm
            state.append((j,np.mean(s),np.std(s)))
        yield k, state

################################################################################
# 临界判断
################################################################################

async def threshHold(qubit,measure,modulation=True):
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    if modulation:
        pilen = qubit.pi_len
        await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
        awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    
        await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
        pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        time.sleep(2)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0],ch_B[:,0]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield j, s

################################################################################
# 优化读出振幅
################################################################################

async def readpowerOpt(qubit,measure,com,att):
    att_setup = Att_Setup(measure,com)
    n = len(measure.delta)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in att:
        att_setup.Att(i,False)
        job = Job(threshHold,(qubit,measure,False),auto_save=False,no_bar=True)
        st, s_st = await job.done()
        x, y = s_st[0], s_st[1]
        offmean, onmean = np.mean(x), np.mean(y)
        offstd, onstd = np.std(x), np.std(y)
        doff, don = np.abs(x-offmean), np.abs(y-onmean)
        popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
        yield [i]*n, [popoff]*n, [popon]*n
    att_setup.close()

################################################################################
# T1
################################################################################

async def T1(qubit,measure,t_rabi,comwave=False):
    name = ''.join((qubit.inst['ex_awg'],'coherence_T1'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.T1_sequence(measure,name,awg,t_rabi,qubit.pi_len)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    await cw.couldRun(awg)

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

################################################################################
# Ramsey
################################################################################

async def Ramsey(qubit,measure,t_rabi,comwave=False):
    name = ''.join((qubit.inst['ex_awg'],'coherence'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Coherence_sequence(measure,name,awg,t_rabi,qubit.pi_len,n_wave=0,seqtype='PDD')
        # await cw.Ramsey_sequence(measure,name,awg,t_rabi,qubit.pi_len)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    await cw.couldRun(awg)

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

################################################################################
# SpinEcho
################################################################################

async def SpinEcho(qubit,measure,t_rabi,len_data,n_wave=0,seqtype='CPMG',comwave=False):
    t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence'))
    t = np.array([t]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Coherence_sequence(measure,name,awg,t_rabi,qubit.pi_len,n_wave,seqtype)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    # await awg.use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    # await awg.write('*WAI')
    # await awg.output_on(ch=qubit.inst['ex_ch'][0])
    # await awg.output_on(ch=qubit.inst['ex_ch'][1])
    # await awg.run()
    # await awg.query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

################################################################################
# AWG crosstalk_cali
################################################################################

async def Z_cross(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave=False):
    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_z.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    ex_awg, z_awg = measure.awg[qubit_ex.inst['ex_awg']], measure.awg[qubit_z.inst['z_awg']]

    await cw.create_wavelist(measure,name_z,(qubit_z.inst['z_awg'],['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,z_awg,name_z)
    await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,ex_awg,name_ex)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Z_cross_sequence(measure,name_z,name_ex,qubit_z.inst['z_awg'],qubit_ex.inst['ex_awg'],v_rabi,qubit_ex.pi_len)
    await cw.awgchmanage(ex_awg,name_ex,qubit_ex.inst['ex_ch'])
    await cw.awgchmanage(z_awg,name_z,qubit_z.inst['z_ch'])
    
    # await measure.awg[qubit_ex.inst['ex_awg']].use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
    # await measure.awg[qubit_ex.inst['ex_awg']].write('*WAI')
    # await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][0])
    # await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][1])
    # await measure.awg[qubit_z.inst['z_awg']].use_sequence(name_z,channels=[qubit_z.inst['z_ch']])
    # await measure.awg[qubit_z.inst['z_awg']].write('*WAI')
    # await measure.awg[qubit_z.inst['z_awg']].output_on(ch=qubit_z.inst['z_ch'])
    # await measure.awg[qubit_z.inst['z_awg']].run()
    # await measure.awg[qubit_ex.inst['ex_awg']].run()
    # await measure.awg[qubit_z.inst['z_awg']].query('*OPC?')
    # await measure.awg[qubit_ex.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

################################################################################
# spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base

async def crosstalkSpec(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave):
    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_ex.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    ex_awg, zz_awg, exz_awg = measure.awg[qubit_ex.inst['ex_awg']], measure.awg[qubit_z.inst['z_awg']]\
        , measure.awg[qubit_ex.inst['z_awg']]
    await cw.create_wavelist(measure,name_z,(qubit_ex.inst['z_awg'],['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,exz_awg,name_z)
    await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,ex_awg,name_ex)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    await cw.genwaveform(measure,zz_awg,[''.join((qubit_z.q_name,'_z'))],qubit_z.inst['z_ch'])
    # await zz_awg.create_waveform(name=''.join((qubit_z.q_name,'_z')), length=len(t_list), format=None)
    # await zz_awg.use_waveform(name=''.join((qubit_z.q_name,'_z')),ch=qubit_z.inst['z_ch'])
    # await zz_awg.output_on(ch=qubit_z.inst['z_ch'])
    if comwave:
        await cw.Speccrosstalk_sequence(measure,name_z,name_ex,exz_awg,ex_awg,v_rabi,400)

    await cw.awgchmanage(ex_awg,name_ex,qubit_ex.inst['ex_ch'])
    await cw.awgchmanage(exz_awg,name_z,qubit_ex.inst['z_ch'])
    
    # await ex_awg.use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
    # await ex_awg.write('*WAI')
    # await ex_awg.output_on(ch=qubit_ex.inst['ex_ch'][0])
    # await ex_awg.output_on(ch=qubit_ex.inst['ex_ch'][1])
    # await exz_awg.use_sequence(name_z,channels=[qubit_ex.inst['z_ch']])
    # await exz_awg.write('*WAI')
    # await exz_awg.output_on(ch=qubit_ex.inst['z_ch'])
    # await exz_awg.run()
    # await ex_awg.run()
    # await exz_awg.query('*OPC?')
    # await ex_awg.query('*OPC?')
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in np.linspace(-1,1,11):
        await zz_awg.stop()
        await cw.zWave(measure,zz_awg,[''.join((qubit_z.q_name,'_z'))],qubit_z.inst['z_ch'],i,during=1100e-9,shift=200*1e-9)
        await zz_awg.run()
        await cw.couldRun(zz_awg)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=500,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
# Randomized Benchmarking
################################################################################

async def RB(qubit_ex,measure,mlist,len_data,gate=None,comwave=False):
    awg = measure.awg[qubit_ex.inst['ex_awg']]
    name_ex = ''.join((qubit_ex.inst['ex_awg'],'rb'))
    for j in tqdm(mlist,desc='RB'):
        await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len_data,len(measure.t_list)))
        await cw.genSeq(measure,awg,name_ex)
        if comwave:
            await cw.rb_sequence(measure,awg,name_ex,j,gate,qubit_ex.pi_len)
        await awg.use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
        await awg.query('*OPC?')
        await cw.readSeq(measure,measure.awg['awgread'],'Read')
        await awg.output_on(ch=qubit_ex.inst['ex_ch'][0])
        await awg.output_on(ch=qubit_ex.inst['ex_ch'][1])
        await cw.couldRun(awg)
        await measure.psg['psg_lo'].setValue('Output','ON')
        await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
        await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
        s = []
        for i in range(3000):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            s.append((Am + Bm)[:,0])

        yield j, np.array(s)

################################################################################
# Randomized Benchmarking waveform
################################################################################

async def RB_waveform(qubit,measure,mlist,len_data,which,DRAGScaling=None,gate=None):
    
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for j in tqdm(mlist,desc='RB'):
        pop = []
        for i in range(len_data):
            pulse = await cw.rbWave(measure,j,gate,qubit.pi_len*1e-9,DRAGScaling=DRAGScaling)
            await cw.writeWave(awg,qname,pulse)
            await cw.couldRun(awg)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            ss = Am + Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict(d)
            pop.append(list(y).count(which)/len(y))
        yield [j], pop
    
################################################################################
# Tomo
################################################################################

async def tomo(qubit,measure,t_rabi,which,DRAGScaling):
    pilen = qubit.pi_len
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(measure,awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in t_rabi:
        pop = []
        for axis in ['Y','X','Z']:
            await cw.tomoTest(measure,awg,i,pilen,axis,qname,DRAGScaling)  
            await cw.couldRun(awg)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            ss = Am + Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict(d)
            pop.append(list(y).count(which)/len(y))
        yield [i], pop
    
        