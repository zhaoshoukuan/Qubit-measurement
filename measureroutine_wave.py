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
import numpy as np, sympy as sy, serial, time, datetime
from easydl import clear_output
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF
from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab import computewave_wave as cww, optimize as op
import pandas as pd
import asyncio

t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
# qubit
################################################################################

class qubit():
    def __init__(self,**kws):
        attribute = ['q_name','inst','T_bias','specfunc','specfunc_cavity','bias','zpulse','f_lo','delta','f_ex','delta_ex','alpha',\
             'power_ex','power_rabi','pi_len','T1','state','timing','envelopename','nwave','readamp','amp',\
             'seqtype','detune','shift','phase','phaseDiff','DRAGScaling','volt','offset','vpp']
        for j in attribute:
            self.__setattr__(j,None)
        if len(kws) != 0:
            for i in kws:
                self.__setattr__(i,kws[i])
    def asdict(self):
        return self.__dict__

    def replace(self,**kws):
        for i in kws:
            # if self.__getattribute__(i):
            if hasattr(self,i):
                self.__setattr__(i,kws[i])
            else:
                raise(f'atrribute {i} is not existed')

class common():
    def __init__(self,freqall,ats,dc,psg,awg,attinst,jpa,qubits={}):
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = {i.q_name:i for i in qubits} 
        self.wave = {}
        self.attinst = attinst
        self.com_lst = [f'com{i}' for i in np.arange(3,15)]
        self.inststate = 0
        self.t_list = np.linspace(0,100000,250000)
        self.t_range = (-90e-6,10e-6)
        self.sample_rate = 2.5e9
        self.readamp = 0.2
        
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
    
    def __init__(self,measure=None,com='com7'):
        if measure == None:
            measure = simpleclass()
            measure.att = {}
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
# 并行处理
################################################################################

def concurrence(task):
    loop = asyncio.get_event_loop()
    for i in task:
        loop.run_until_complete(asyncio.wait(i))
    # loop.close()


################################################################################
# AWG恢复设置
################################################################################

async def resetAwg(awg):
    for i in awg:  
        await awg[i].setValue('Run Mode','Triggered')   # Triggered, Continuous
        # if i == 'awg_trig':
        #     await cw.genwaveform(awg[i],['Readout_Q'],[5])
        #     await cw.couldRun(awg[i])
        #     # await awg[i].write('TRIGg:SOUR %s'%'INT')   
        #     await awg[i].write('TRIGGER:INTERVAL %f'%260e-6) 
        #     # for m in range(8):
        #     #     await awg[i].write('SOUR%d:TINP %s'%((m+1),'ATR'))
        for j in range(8):
            await cww.genwaveform(awg[i],[f'ch{j+1}'],[j+1])
            await awg[i].setValue('Vpp',1.5,ch=j+1)
            await awg[i].setValue('Offset',0,ch=j+1)
            await awg[i].write('OUTPUT%d:WVALUE:ANALOG:STATE %s'%(j+1,'FIRST'))  #FIRST, ZERO
            await awg[i].write('SOUR%d:TINP %s'%((j+1),'ATR'))        
            for k in range(4):
                await awg[i].write('OUTPUT%d:WVALUE:MARKER%d %s'%(j+1,(k+1),'FIRST'))            #FIRST, LOW, HIGH

################################################################################
# AWG同步
################################################################################

async def awgSync(measure):
    t_list = measure.t_list
    wf = WF(t_list)
    awg_list = measure.awg
    for i in awg_list:
        wait = 'ATR' if awg_list[i] == awg_list['awg_trig'] else 'ATR'
        await awg_list[i].create_waveform(name=''.join((i,'_sync1')),length=len(measure.t_list),format=None)
        await awg_list[i].create_sequence(name=''.join((i,'_syncSeq')),steps=3,tracks=1)
        height,t_end,width = [1],[50000],[2000]
        sample = wf.square_wave(t_end,width,height)
        await awg_list[i].update_waveform(sample,name=''.join((i,'_sync1')))
        for j in range(3):
            j += 1
            goto = 'FIRST' if j == 3 else 'NEXT'
            await awg_list[i].set_sequence_step(name=''.join((i,'_syncSeq')),sub_name=[''.join((i,'_sync1'))],\
                                                   step=j,wait=wait,goto=goto,repeat=1,jump=None)
        await awg_list[i].use_sequence(name=''.join((i,'_syncSeq')),channels=[1])
        await awg_list[i].query('*OPC?')
        await awg_list[i].output_on(ch=1)
        await awg_list[i].setValue('Force Jump',1,ch=1)
        await awg_list[i].run()  

################################################################################
# 等效磁通计算
################################################################################

def voltTophi(qubit,bias,distance=0,freq=0):
    tm = np.arange(-qubit.T_bias[0]/2,0,0.001)
    t = tm + qubit.T_bias[1] if qubit.T_bias[1] > bias else -tm + qubit.T_bias[1]
    x, func = sy.Symbol('x'), qubit.specfunc
    y = sy.lambdify(x,func,'numpy')
    spec = y(t)
    bias_deviate = t[np.abs(spec-freq).argmin()] - bias
    f_bias = y(bias)
    f_distance = y(bias+distance)
    f_deviate = f_distance - f_bias
    return f_bias, f_deviate, bias_deviate

################################################################################
# 腔频率设置
################################################################################

async def resn(f_cavity):
    f_lo = f_cavity.max() + 50e6
    delta =  f_lo - f_cavity 
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
# 直流源设置
################################################################################

async def dcManage(measure,dcstate={},readstate=[],calimatrix=None,qnum=10,readamp=0.3):
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    fread = []
    qubitToread = []
    for i,j in enumerate(measure.freqall):
        bias[i] = dcstate[j] if j in dcstate else 0
        if readstate != None:
            if readstate == []:
                if j in dcstate:
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
            else:
                if j in readstate: 
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
    if readstate != None:
        measure.qubitToread = qubitToread
        f_lo, delta, n = await resn(np.array(fread))
        measure.n, measure.delta, measure.f_lo = n, delta, f_lo
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
        await cww.couldRun(measure.awg['awgread'])
    current = matrix.I * np.mat(bias).T
    for i,j in enumerate(measure.freqall):
        await measure.dc[j].DC(round(current[i,0],3))

    # return f_lo, delta, n

################################################################################
# 激励频率计算
################################################################################

async def exMixing(f):
    if f == {}:
        return 
    qname = [i for i in f]
    f_ex = np.array([f[i] for i in f])
    ex_lo = f_ex.max() + 100e6
    delta =  ex_lo - f_ex
    delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
    # n = len(f_ex)
    return ex_lo, delta_ex

################################################################################
# 激励源设置
################################################################################

async def exManage(measure,exstate=[],qnum=10):
    qubits = measure.qubits
    f_ex1, f_ex2, f_ex3 = {}, {}, {}
    delta_ex1, delta_ex2, delta_ex3 = {}, {}, {}
    for i,j in enumerate(qubits):
        if j in exstate:
            if qubits[j].inst['ex_lo'] == 'psg_ex1':
                f_ex1[j] = qubits[j].f_ex
            if qubits[j].inst['ex_lo'] == 'psg_ex2':
                f_ex2[j] = qubits[j].f_ex
            if qubits[j].inst['ex_lo'] == 'psg_ex3':
                f_ex3[j] = qubits[j].f_ex
    if f_ex1 != {}:
        ex_lo1, delta_ex1 = await exMixing(f_ex1)
        await measure.psg['psg_ex1'].setValue('Frequency',ex_lo1)
        await measure.psg['psg_ex1'].setValue('Output','ON')
    if f_ex2 != {}:
        ex_lo2, delta_ex2 = await exMixing(f_ex2)
        await measure.psg['psg_ex2'].setValue('Frequency',ex_lo2)
        await measure.psg['psg_ex2'].setValue('Output','ON')
    if f_ex3 != {}:
        ex_lo3, delta_ex3 = await exMixing(f_ex3)
        await measure.psg['psg_ex3'].setValue('Frequency',ex_lo3)
        await measure.psg['psg_ex3'].setValue('Output','ON')
    delta_ex = {**delta_ex1,**delta_ex2,**delta_ex3}

    return delta_ex

async def zManage(measure,dcstate={},calimatrix=None,qnum=10):
    qubits = measure.qubits
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    for i,j in enumerate(qubits):
        bias[i] = dcstate[j] if j in dcstate else 0
        
    current = matrix.I * np.mat(bias).T
    current = {f'q{i+1}':current[i,0] for i in range(qnum)}
    return current

################################################################################
# 执行激励seq并行处理
################################################################################

async def executeEXseq(measure,update_seq,len_data,comwave,exstate=[],**paras):
    delta_ex = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    task1, task2, namelist, awglist = [], [], [], []
    for i in delta_ex:
        qubit = bit[i]
        qubit.delta_ex = delta_ex[i]
        # kind = ''.join((qubit.inst['ex_awg'],qubit.q_name,'_ex'))
        kind = ''.join((qubit.q_name,'_ex'))
        awg = measure.awg[qubit.inst['ex_awg']]
        # namelist.append(kind)
        # awglist.append(awg)
        await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
        if comwave:
            task_ready = update_seq(qubit,measure,kind,**paras)
            task1.append(task_ready)
        task_manageawg = cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])
        task2.append(task_manageawg)
    return [task1,task2]

################################################################################
# 执行zseq并行处理
################################################################################

async def executeZseq(measure,update_seq,len_data,comwave,dcstate={},calimatrix=None,**paras):
    current = await zManage(measure,dcstate,calimatrix=calimatrix)
    bit = measure.qubits
    task1, task2, namelist, awglist = [], [], [], []
    for i in current:
        qubit = bit[i]
        # kind = ''.join((qubit.inst['z_awg'],qubit.q_name,'_z'))
        kind = ''.join((qubit.q_name,'_z'))
        awg = measure.awg[qubit.inst['z_awg']]
        await cww.gen_packSeq(measure,kind,awg,['step'],len_data)
        if comwave:
            if 'volt' in paras:
                task_ready = update_seq(qubit,measure,kind,**paras)
            else:
                task_ready = update_seq(qubit,measure,kind,volt=current[i],**paras)
            task1.append(task_ready)
        task_manageawg = cww.awgchmanage(awg,kind,qubit.inst['z_ch'])
        task2.append(task_manageawg)
    return [task1,task2]
    # concurrence([task1,task2])

################################################################################
# 执行激励并行处理
################################################################################

async def executeEXwave(measure,update_wave,exstate=[],**paras):
    delta_ex = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    task1, task2, namelist, awglist = [], [], [], []
    for i in delta_ex:
        qubit = bit[i]
        qubit.delta_ex = delta_ex[i]
        namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
        chlist = qubit.inst['ex_ch']
        awg = measure.awg[qubit.inst['ex_awg']]
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        task_ready = cww.writeWave(awg,name=namelist,pulse=pulselist)
        task1.append(task_ready)
        task_manageawg = cww.couldRun(awg,chlist,namelist)
        task2.append(task_manageawg)
    return [task1,task2]

################################################################################
# 执行z并行处理
################################################################################

async def executeZwave(measure,update_wave,dcstate={},calimatrix=None,**paras):
    current = await zManage(measure,dcstate=dcstate,calimatrix=calimatrix)
    bit = measure.qubits
    task1, task2, namelist, awglist = [], [], [], []
    for i in current:
        qubit = bit[i]
        kind = ''.join((qubit.inst['z_awg'],qubit.q_name,'_z'))
        awg = measure.awg[qubit.inst['z_awg']]
        if 'volt' in paras:
            pulselist = await cww.funcarg(update_wave,qubit,**paras)
        else:
            pulselist = await cww.funcarg(update_wave,qubit,volt=current[i],**paras)
        task_ready = cww.writeWave(awg,name=namelist,pulse=pulselist)
        task1.append(task_ready)
        task_manageawg = cww.couldRun(awg,chlist,namelist)
        task2.append(task_manageawg)
    return [task1,task2]

################################################################################
# 开关JPA
################################################################################

async def jpa_switch(measure,state='OFF'):
    if state == 'ON':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','ON')
        await measure.psg[measure.jpa.inst['pump']].setValue('Frequency',(measure.jpa.f_ex))
        await measure.psg[measure.jpa.inst['pump']].setValue('Power',measure.jpa.power_ex)
        await measure.dc[measure.jpa.q_name].DC(measure.jpa.bias)
    if state == 'OFF':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','OFF')
        await  measure.dc[measure.jpa.q_name].DC(0)

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
            if 'psg' in i:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                Mform = await inst[i].getValue('Mform')
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output.strip('\n\r'),\
                    'moutput':Moutput.strip('\n\r'),'mform':Mform.strip('\n\r'),'error':err[0]}
                state[i] = sm
            else:
                current = await inst[i].getValue('Offset')
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n\r'))  
                load = 'high Z' if load != 50 else 50
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'offset':current,'load':load,'error':err[0]}
                state[i] = sm
            # if 'awg' in i:
            #     runstate = await inst[i].getValue('Offset')
            #     load = await inst[i].getValue('Load')
            #     load = eval((load).strip('\n\r'))  
            #     load = 'high Z' if load != 50 else 50
            #     err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
            #     sm = {'offset':current,'load':load,'error':err[0]}
            #     state[i] = sm
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
            measure.wave = {}

################################################################################
# 恢复仪器最近状态
################################################################################

async def RecoverInst(measure,state=None):
    if state is None:
        state = measure.inststate
    for i in state:
        if 'psg' in i:
            await measure.psg[i].setValue('Frequency',eval(state[i]['freq'].strip('GHz'))*1e9)
            await measure.psg[i].setValue('Power',eval(state[i]['power'].strip('dBm')))
            if state[i]['output'] == '1':
                await measure.psg[i].setValue('Output','ON')
            else:
                await measure.psg[i].setValue('Output','OFF')
        else:
            await measure.dc[i].DC(state[i]['offset'])

################################################################################
# S21
################################################################################

async def S21(qubit,measure,modulation=False,f_lo=None,f=None):
    #await jpa_switch(measure,state='OFF')
    if f_lo == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-3,3,121)*1e6 + f_lo
    else:
        freq = np.linspace(-3,3,121)*1e6 + f_lo
        delta, n = measure.delta, measure.n
    if modulation:
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
    await cww.couldRun(measure.awg['awgread'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    if f is not None:
        freq = f
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield i-delta, s

async def test(measure,n):
    for i in range(n):
        time.sleep(0.1)
        # await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 读出信号相位噪声
################################################################################

async def rPhase(measure,phase):
    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in phase:
        await cww.modulation_read(measure,measure.delta,readlen=1200,phase=i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False,flo=None,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,121)*1e6+f_lo
    for i in measure.psg:
        if i != 'psg_lo' and i != 'psg_pump':
            await measure.psg[i].setValue('Output','OFF')
    length = len(freq) if freq is not None else 121
    job = Job(S21, (qubit,measure,modulation,flo,freq),auto_save=True,max=length,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cww.ats_setup(measure.ats,delta)
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
        base = 0
        for i in range(15):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 15
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base,f_s21, s_s21

################################################################################
# S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,modulation=False):
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        # await measure.dc['q2'].DC(5)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsFlux_awgoffset
################################################################################

async def S21vsFlux_awgoffset(qubit,measure,current,calimatrix,modulation=False):
    awg = measure.awg[qubit.inst['z_awg']]
    zname = [''.join((qubit.q_name,'_z'))]
    await cww.genwaveform(awg,zname,qubit.inst['z_ch'])
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        pulse = await cww.funcarg(cww.zWave,qubit,pi_len=5000e-9,offset=i,shift=-3000e-9)
        await cww.writeWave(awg,zname,pulse,mark=False)
        await cww.couldRun(awg)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att,com='com8'):
    l = measure.readlen
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=None)
    for i in att:
        # await measure.attinst['com8'].set_att(i)
        measur.readamp = i
        await cww.modulation_read(measure,measure.delta,readlen=2000)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    measure.readlen = l

################################################################################
# SingleSpec
################################################################################

async def singlespec(qubit,measure,freq,modulation=False,f_read=None,readponit=True):
    if readponit:
        f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,f_read)
    else:
        n, base = measure.n, measure.base
    await measure.psg['psg_trans'].setValue('Output','ON')
    await measure.psg['psg_trans'].setValue('Moutput','ON')
    for i in freq:
        await measure.psg['psg_trans'].setValue('Frequency',i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base
    await measure.psg['psg_trans'].setValue('Output','OFF')
################################################################################
# SingleSpec扫电压
################################################################################

async def specbias(qubit,measure,ftarget,bias,modulation=False):
    await measure.dc[qubit.q_name].DC(round(np.mean(bias),3))
    await measure.psg['psg_trans'].setValue('Frequency',ftarget)
    f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    await measure.psg['psg_trans'].setValue('Output','ON')
    for i in bias:
        await measure.dc[qubit.q_name].DC(i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base
    await measure.psg['psg_trans'].setValue('Output','OFF')

################################################################################
# Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,modulation=False):
    current = np.linspace(-qubit.T_bias[0]*0.35,qubit.T_bias[0]*0.35,36) + qubit.T_bias[1] 
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        # await measure.dc[qubit.q_name].DC(i)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,True),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Spec2d_awg
################################################################################

async def spec2d_awg(qubit,measure,freq,calimatrix,modulation=False):
    # current = np.linspace(-qubit.T_bias[0]*0.35,qubit.T_bias[0]*0.35,36) + qubit.T_bias[1] 
    current = np.linspace(0,1,51)
    namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    chlist = qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    await cww.genwaveform(z_awg,namelist,chlist)
    await z_awg.setValue('Offset',-0.0,ch=chlist[0])
    # await dcManage(measure,dcstate=None,readstate=measure.qubitToread,calimatrix=calimatrix)
    # f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    
    for i in current:
        delta_ex, flux = await exManage(measure,dcstate={qubit.q_name:i},exstate={},calimatrix=calimatrix)
        # await cw.zWave(z_awg,zname,during=17000e-9,volt=flux[qubit.q_name],shift=1150e-9)
        # await cw.couldRun(z_awg)
        pulselist = await cww.funcarg(cww.rabiWave,qubit,pi_len=17000e-9,amp=flux[qubit.q_name])
        await cww.writeWave(z_awg,name=namelist,pulse=pulselist[:1])
        await cww.couldRun(z_awg,chlist)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,False),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# RabiPower
################################################################################

async def rabiPower(measure,amp,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in amp:
        pulselist = await cww.funcarg(cww.rabiWave,qubit,amp=i)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# RabiTime
################################################################################

async def rabiTime(measure,t_rabi,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_rabi:
        pulselist = await cww.funcarg(cww.rabiWave,qubit,pi_len=i/1e9)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# RabiTime_seq
################################################################################

async def rabiTime_seq(measure,t_rabi,exstate=[],comwave=True):

    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_ex'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]

    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
    if comwave:
        await cww.Rabi_sequence(qubit,measure,kind,t_rabi)
    await cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# RabiPower_seq
################################################################################

async def rabiPower_seq(measure,amp,exstate=[],comwave=True):

    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_ex'))
    len_data, t = len(amp), np.array([amp]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]

    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
    if comwave:
        await cww.RabiPower_sequence(qubit,measure,kind,amp)
    await cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# RabiPower_many
################################################################################

async def rabiPower_many(measure,amp,exstate=[],comwave=True):

    len_data, t = len(amp), np.array([amp]*measure.n).T
    task = await executeEXseq(measure,cww.Rabi_sequence,len_data,comwave,exstate=exstate,t_rabi=amp)
    concurrence(task)

    await measure.psg['psg_lo'].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 优化pi脉冲
################################################################################
    
async def pipulseOpt(measure,nwave,wavlen,optwhich='t',exstate=[]):
    qubit = measure.qubits[exstate[0]]
    if optwhich == 't':
        pilen = qubit.pi_len
        start = 0.5*pilen if 0.5*pilen > 10 else 1
        end = 2.5*pilen if 0.5*pilen > 10 else pilen+30
        x = np.linspace(start,end,wavlen)
        func = rabiTime_seq
    if optwhich == 'amp':
        amp = qubit.amp
        start = 0.5*amp
        end = 1.5*amp if 1.5*amp <=1 else 1
        x = np.linspace(start,end,wavlen)
        func = rabiPower_seq
    for i in range(nwave):
        qubit.nwave = 4*i+1
        job = Job(func, (measure,x,exstate), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [4*i+1]*measure.n, t_r, s_r
    qubit.nwave = 1

# ################################################################################
# # 优化pi脉冲
# ################################################################################
    
# async def pipulseOpt(measure,nwave,wavlen,exstate=[]):
#     qubit = measure.qubits[exstate[0]]
#     pilen = qubit.pi_len
#     start = 0.5*pilen if 0.5*pilen > 10 else 1
#     end = 2.5*pilen if 0.5*pilen > 10 else pilen+30
#     t = np.linspace(start,end,wavlen)
#     for i in range(nwave):
#         qubit.nwave = 2*i+1
#         job = Job(rabiTime_seq, (measure,t,exstate), max=500,avg=True,auto_save=False)
#         t_r, s_r = await job.done()
#         yield [2*i+1]*measure.n, t_r, s_r

################################################################################
# Ramsey_seq
################################################################################

async def Ramsey_seq(measure,t_Ramsey,exstate=[],comwave=True):

    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_ex'))
    len_data, t = len(t_Ramsey), np.array([t_Ramsey]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]

    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
    if comwave:
        await cww.Coherence_sequence(qubit,measure,kind,t_Ramsey)
    await cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# SpinEcho_seq
################################################################################

async def SpinEcho_seq(measure,t_Ramsey,exstate=[],comwave=True):
    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_ex'))
    len_data, t = len(t_Ramsey), np.array([t_Ramsey]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]

    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
    if comwave:
        await cww.Coherence_sequence(qubit,measure,kind,t_Ramsey)
    await cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 优化读出点
################################################################################

async def readOp(measure,exstate=[]):

    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    concurrence(task)

    await cww.couldRun(measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    for j, i in enumerate(['OFF','ON']):
        for k in exstate:
            await measure.psg[measure.qubits[k].inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False,max=121)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

# async def readOp(measure,exstate=[]):


#     qubit = measure.qubits[exstate[0]]
#     namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
#     chlist = qubit.inst['ex_ch']
#     awg_ex = measure.awg[qubit.inst['ex_awg']]

#     await cww.genwaveform(awg_ex,namelist,chlist)
#     pulselist = await cww.funcarg(cww.rabiWave,qubit)
#     await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
#     await cww.couldRun(awg_ex,chlist)
#     await cww.couldRun(measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
#     for j, i in enumerate(['OFF','ON']):
#         await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
#         job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False,max=121)
#         f_s21, s_s21 = await job.done()
#         n = np.shape(s_s21)[1]
#         yield [j]*n, f_s21, s_s21


################################################################################
# 对比度
################################################################################

async def visibility(n,s0,s1):
    theta = np.arange(0, 2*np.pi, 0.01)
    data = []
    for i in range(n):
        c0, c1 = np.mean(s0), np.mean(s1)
        s0 = s0 / ((c1-c0)/np.abs(c1-c0))
        s1 = s1 / ((c1-c0)/np.abs(c1-c0))
        s0 = np.real(s0)
        s1 = np.real(s1)
        bins = np.linspace(np.min(np.r_[s0,s1]), np.max(np.r_[s0,s1]), 61)
        y0,_ = np.histogram(s0, bins=bins)
        y1,_ = np.histogram(s1, bins=bins)
        inte0 = np.cumsum(y0)/np.sum(y0)
        inte1 = np.cumsum(y1)/np.sum(y0)
        inte_diff = np.cumsum(y0)/np.sum(y0) - np.cumsum(y1)/np.sum(y1)
        offstd, onstd = np.std(s0), np.std(s1)
        roff = np.real(c0) + offstd * np.cos(theta)
        ioff = np.imag(c0) + offstd * np.sin(theta)
        ron = np.real(c1) + onstd * np.cos(theta)
        ion = np.imag(c1) + onstd * np.sin(theta)
        data.append([inte0,inte1,inte_diff,(roff,ioff),(ron,ion)])
    return data

################################################################################
# 临界判断
################################################################################

async def threshHold(measure,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    await cww.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=3000)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        time.sleep(0.5)
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        s = Am + 1j*Bm
        sq = I + 1j*Q
        yield j, s, sq

################################################################################
# 优化读出功率
################################################################################

async def readpowerOpt(measure,which,readamp):
    # qubit = measure.qubits[exstate[0]]
    n = len(measure.delta)
    for k in readamp:
        measure.readamp = k
        await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=5000)
        await cww.couldRun(measure.awg['awgread'])
        
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        ss = Am + 1j*Bm
        d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(n)]
        y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
        # d = list(zip(np.real(ss[:,0]),np.imag(ss[:,0])))
        # y = measure.predict[measure.qubitToread[0]](d)
        # pop = list(y).count(which[0])/len(y)
        yield [k]*n, pop


        # S = list(s_off) + list(s_on)
        # x,z = np.real(S), np.imag(S)
        # d = list(zip(x,z))
        # y = measure.predict[qubit.q_name](d)
        # pop = list(y).count(1)/len(y)
        # yield [i]*n, [pop]

    #     offmean, onmean = np.mean(x), np.mean(y)
    #     offstd, onstd = np.std(x), np.std(y)
    #     # doff, don = np.abs(x-offmean), np.abs(y-onmean)
    #     d = np.abs(offmean-onmean)
    #     # popoff, popon = d/offstd, d/onstd
    #     snr = 2*d**2/(offstd+onstd)**2
    #     # popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
    #     # yield [i]*n, [popoff]*n, [popon]*n
    #     yield [i]*n, [snr]*n
    # att_setup.close()

################################################################################
# AllXY drag detune
################################################################################
    
async def AllXYdragdetune(measure,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    coef = np.linspace(-3,3,61)

    qubit = measure.qubits[exstate[0]]
    alpha = qubit.alpha*2*np.pi
    ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]

    # await cw.modulation_read(measure,measure.delta,readlen=measure.readlen)
    # await cw.genwaveform(ex_awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for j in [['X','Yhalf'],['Y','Xhalf']]:
        for i in coef:
            pulse1 = await cww.funcarg(cww.singleQgate,qubit,axis=j[0],DRAGScaling=i/alpha)
            pulse2 = await cww.funcarg(cww.singleQgate,qubit,axis=j[1],DRAGScaling=i/alpha)
            pulse = np.array(pulse1) + np.array(pulse2)
            await cww.writeWave(ex_awg,ex_name,pulse)
            await cww.couldRun(ex_awg,namelist=ex_name,chlist=ex_ch)
            ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=True)
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)

            # ss = Am + 1j*Bm
            # d = list(zip(np.real(ss),np.imag(ss)))
            # y = measure.predict[qubit.q_name](d)
            # pop = list(y).count(which)/len(y)
            # yield [i], [pop]

            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield [i], s-measure.base

################################################################################
# T1
################################################################################

async def T1(measure,shift,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in shift:
        pulselist = await cww.funcarg(cww.rabiWave,qubit,shift=i/1e9)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# T1_seq
################################################################################

async def T1_seq(measure,t_T1,exstate=[],comwave=True):

    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_ex'))
    len_data, t = len(t_T1), np.array([t_T1]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]

    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data)
    if comwave:
        await cww.T1_sequence(qubit,measure,kind,t_T1)
    await cww.awgchmanage(awg,kind,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 真空拉比
################################################################################

async def vRabi(measure,t_rabi,v_rabi,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    kind = ''.join((qubit.inst['ex_awg'],'_z'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['z_awg']]

    await cww.gen_packSeq(measure,kind,awg,['step'],len_data)

    # if clearseq:
    #     await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    #     measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    #     await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])
    # zseqname, z_awg = qubit.inst['z_awg']+'_vrabi', measure.awg[qubit.inst['z_awg']]
    # zname = [qubit.q_name+'_z']
    # await cw.create_wavelist(measure,zseqname,(z_awg,['step'],len(t_rabi),len(measure.t_list)))
    # await cw.genSeq(measure,z_awg,zseqname)
    await cww.awgchmanage(awg,kind,qubit.inst['z_ch'])
    
    for i, j in enumerate(v_rabi):
        await cww.Z_sequence(measure,qubit,kind,t_rabi,volt=j)
        await cww.couldRun(z_awg)
        comwave = True if i == 0 else False
        job = Job(T1_seq, (measure,(t_rabi+500),exstate,comwave), tags=(exstate+['vRabi']), max=500,avg=True, no_bar=True)
        t_t, s_t = await job.done()
        # clear_output()
        yield [j]*measure.n, t_t, s_t


################################################################################
# AC-Stark（XY与Readout时序矫正)
################################################################################

async def singleacStark(measure,t_shift,exstate=[],comwave=True):

    await cww.ac_stark_wave(measure)
    job = Job(T1_seq, (measure,t_shift,exstate,comwave), tags=(exstate+['acstark']), max=500,avg=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def acStark(measure,t_T1,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    freq = np.linspace(-50,50,51)*1e6 + qubit.f_ex + qubit.delta_ex

    # await cww.ac_stark_wave(measure)
    # await cww.couldRun(measure.awg['awgread'])

    for i,j in enumerate(freq):
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',j)
        comwave = True if i == 0 else False
        job = Job(singleacStark, (measure,t_T1,exstate,comwave), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_ac = await job.done()
        yield [j-qubit.delta_ex]*measure.n, t_shift, s_ac

################################################################################
# Z_Pulse（Z与Readout时序矫正)
################################################################################

async def singleZpulse(measure,t_shift,exstate=[],comwave=True,updatez=True):
    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    if updatez:
        pulse = await cww.funcarg(cww.zWave,qubit,pi_len=1000/1e9,volt=0.05,shift=3000e-9)
        await cww.writeWave(z_awg,zname,pulse,mark=False)
    await cww.couldRun(z_awg)

    job = Job(T1_seq, (measure,t_shift,exstate,comwave), tags=(exstate+['z_pulse']), max=500,avg=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def zPulse(measure,t_T1,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,pi_len=1000/1e9,volt=0.01,shift=3000e-9)
    volt = np.linspace(-0.01,0,51)
    for i,j in enumerate(volt):
        pulse2 = await cww.funcarg(cww.zWave,qubit,pi_len=3000/1e9,volt=j,shift=1500e-9)
        pulse = np.array(pulse1) + np.array(pulse2)
        await cww.writeWave(z_awg,zname,pulse,mark=False)
        await cww.couldRun(z_awg,zch,zname)
        comwave = True if i == 0 else False
        job = Job(singleZpulse, (measure,t_T1,exstate,comwave,False), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_z = await job.done()
        yield [j]*measure.n, t_shift, s_z


async def singleZ(measure,t_T1,which=0,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    for k in t_T1:
            pulselist = await cww.funcarg(cww.rabiWave,qubit,shift=k/1e9)
            await cww.writeWave(ex_awg,name=exname,pulse=pulselist)
            await cww.couldRun(ex_awg,exch)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A,ch_B
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # d = list(zip(np.real(ss),np.imag(ss)))
            d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
            y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
            pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
            pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
            yield [k], pop

async def zPulse_pop(measure,t_T1,which=0,exstate=[]):

    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,pi_len=1000/1e9,volt=0.05,shift=3000e-9)

    volt = np.linspace(-0.05,0,21)

    for j in volt:
        pulse2 = await cww.funcarg(cww.zWave,qubit,pi_len=3000/1e9,volt=j,shift=1500e-9)
        pulse = np.array(pulse1) + np.array(pulse2)
        await cww.writeWave(z_awg,zname,pulse,mark=False)
        await cww.couldRun(z_awg,zch,zname)
        job = Job(singleZ,(measure,t_T1,which,exstate),max=len(t_T1))
        t, pop = await job.done()
        yield [j], t, pop
        
        

################################################################################
# Ramsey
################################################################################

async def Ramsey(measure,t_run,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_run:
        pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# SpinEcho
################################################################################

async def SpinEcho(measure,t_run,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_run:
        pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def crosstalkSpec(measure,v_rabi,dcstate={},comwave=False):
    len_data = len(v_rabi)
    t, name_ex, name_z = np.array([v_rabi]*measure.n).T, ''.join((qubit_ex.inst['ex_awg'],'_ex')), ''.join((qubit_ex.inst['z_awg'],'_z'))
    ex_awg, zz_awg, exz_awg = measure.awg[qubit_ex.inst['ex_awg']], measure.awg[qubit_z.inst['z_awg']]\
        , measure.awg[qubit_ex.inst['z_awg']]

    await cww.gen_packSeq(measure,name_z,exz_awg,['Z'],len_data,readseq=False)
    await cww.gen_packSeq(measure,name_ex,ex_awg,['I','Q'],len_data)
    await cw.genwaveform(zz_awg,[f"ch{qubit.inst['z_ch'][0]}"],qubit_z.inst['z_ch'])
    if comwave:
        await cw.Speccrosstalk_sequence(measure,qubit_ex,name_z,name_ex,v_rabi,400)

    await cw.awgchmanage(ex_awg,name_ex,qubit_ex.inst['ex_ch'])
    await cw.awgchmanage(exz_awg,name_z,qubit_ex.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    for i in np.linspace(-1,1,11):
        await cw.zWave(zz_awg,[f"ch{qubit.inst['z_ch'][0]}"],i,during=1100e-9,shift=200*1e-9)
        await cw.couldRun(zz_awg)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=500,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
# RTO_Notomo
################################################################################

async def RTO_Notomo(measure,t_run,exstate=[]):
    await cww.couldRun(measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_run:
        pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=500/1e9)
        await cww.writeWave(awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base