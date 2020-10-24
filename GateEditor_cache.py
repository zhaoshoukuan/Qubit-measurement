import numpy as np
import functools
from cacheout import Cache
import logging
log = logging.getLogger(__name__)

from .wavedata import A,vIQmixer,Wavedata
from .GateIndex_cache import GateIndex
from .ctrlseq import Blank_cSeq

def RawWave(cfg,gateMap,gateindex=None):
    '''根据索引列表合并门的核心波形，并做初步时序处理'''
    All_Qubits = cfg.get('Chip').get('All_Qubits',[])

    if gateindex is None:
        gateindex = GateIndex(cfg)
    else:
        gateindex.renew(cfg)
    cSeq_dict = gateindex.mix_combine_gateMap(gateMap)
    max_len = max((0,*(cSeq.len for cSeq in cSeq_dict.values())))

    xy_dict,z_dict={},{}
    for qubit in All_Qubits:
        sr = (cfg[qubit]['gate']['sRate']['XY'],cfg[qubit]['gate']['sRate']['Z'])
        default_cSeq_kernal = Blank_cSeq(width=0,sRate=sr)
        cSeq_kernal=cSeq_dict.get(qubit,default_cSeq_kernal)

        xy_dict[qubit] = cSeq_kernal.xyCH.setLen(-max_len)
        z_dict[qubit] = cSeq_kernal.zCH.setLen(-max_len)
    return xy_dict,z_dict

def cali_xyWave_carry(cfg,xy_dict):
    '''xyWave 加载载波'''
    All_Qubits = cfg.get('Chip').get('All_Qubits',[])
    wdLen=cfg.get('ADC_Meas').get('global').get('timeSeq').get('wdLen')
    probT0=cfg.get('ADC_Meas').get('global').get('timeSeq').get('probT0')
    gapTime=cfg.get('ADC_Meas').get('global').get('timeSeq').get('gapTime')
    switchGap=cfg.get('ADC_Meas').get('global').get('timeSeq').get('switchGap')

    new_xy_dict,mask_dict={},{}
    for qubit in All_Qubits:
        carrywave_freq=cfg.get(qubit).get('para').get('drivCarryFreq')
        driv_scale_Q=cfg.get(qubit).get('cali').get('drivIQcali').get('scale_Q',1)
        driv_phase_Q=cfg.get(qubit).get('cali').get('drivIQcali').get('phase_Q',0)
        off_I_wd=cfg.get(qubit).get('cali').get('drivIQcali').get('off_I_wd',0)
        off_Q_wd=cfg.get(qubit).get('cali').get('drivIQcali').get('off_Q_wd',0)

        wd_xy_kernal = xy_dict[qubit]
        delay = probT0-gapTime-wd_xy_kernal.len
        wd_xy_raw = wd_xy_kernal.setLen(wdLen)>>delay
        xy_mask = A.wdMask(wd_xy_raw,extend_len=switchGap)

        xy_final = vIQmixer.carry_wave(carrywave_freq,IQ=wd_xy_raw,phase=0,Q_scale=driv_scale_Q,Q_degree=driv_phase_Q)
        ## 通过写波形 校准IQ混频器的offset，
        ## NOTE: 两个比特共用同一路XY线时，只需把校准参数写入其中一个比特的参数表中，否则会加在一起出错
        new_xy_dict[qubit] = xy_final + off_I_wd + 1j*off_Q_wd
        mask_dict[qubit] = xy_mask
    return new_xy_dict,mask_dict

def cali_zWave_crossTalk(cfg,z_dict):
    '''对Zpulse做串扰校准'''
    crossTalk_Qubits = cfg.get('Chip').get('crossTalk_Qubits',[])
    matrix = cfg.get('Chip').get('crossTalkMatrix_pulse')
    pulseAmpPeriod = np.array([[cfg[qubit]['para']['pulseAmpPeriod']] for qubit in crossTalk_Qubits])
    inv_matrix=np.linalg.inv(matrix)

    z_data_tuple = [z_dict[qubit].data for qubit in crossTalk_Qubits]
    z_data = np.stack(z_data_tuple,axis=0)
    z_data_new = (inv_matrix @ z_data) * pulseAmpPeriod
    sRate_tuple = [z_dict[qubit].sRate for qubit in crossTalk_Qubits]

    for qubit,z_data,z_sRate in zip(crossTalk_Qubits, z_data_new, sRate_tuple):
        z_dict[qubit] = Wavedata(z_data,z_sRate)
    return z_dict

@functools.lru_cache(maxsize=128)
def predistort(waveform, sRate, zCali):
    """Predistort input waveform.
    Parameters
    ----------
        waveform : tuple, 由于要用缓存，传入参数需为可Hash量
            Waveform data to be pre-distorted
        zCali: ((2.55e-9, -28e-3), (8.02e-9, -28e-3), (101.5e-9, -14.5e-3),(369.7e-9,-6.2e-3))
    Returns
    -------
    waveform : complex numpy array
        Pre-distorted waveform
    """
    dt = 1 / sRate
    wf_size = len(waveform)
    # tau1, A1, tau2, A2, tau3, A3, tau4, A4, tau5, A5 = zCali
    # print('tau:', tau1, A1, tau2, A2, tau3, A3, tau4, A4)
    # pad with zeros at end to make sure response has time to go to zero
    # pad_time = 6 * max([tau1, tau2, tau3])
    pad_time = 4e-6  # 这里默认改为增加4us的pad，返回时舍去1us，保留3us拖尾
    pad_size = round(pad_time / dt)
    pad_size_2 = round(3e-6 / dt)  # 保留的点数
    padded_zero = np.zeros(pad_size)
    padded = np.append(waveform,padded_zero)

    Y = np.fft.rfft(padded, norm='ortho')

    omega = 2 * np.pi * np.fft.rfftfreq(wf_size+pad_size, dt)
    # H = (1 + (1j * A1 * omega * tau1) / (1j * omega * tau1 + 1) +
    #      (1j * A2 * omega * tau2) / (1j * omega * tau2 + 1) +
    #      (1j * A3 * omega * tau3) / (1j * omega * tau3 + 1) +
    #      (1j * A4 * omega * tau4) / (1j * omega * tau4 + 1) +
    #      (1j * A5 * omega * tau5) / (1j * omega * tau5 + 1))

    # tau_A_array = np.array(zCali).reshape(-1,2)
    H = 1
    for tau,A in zCali:
        H += (1j * A * omega * tau) / (1j * omega * tau + 1)

    Yc = Y / H
    yc = np.fft.irfft(Yc, norm='ortho')
    # return yc[:wf_size]
    #这里返回增加了3us拖尾的序列
    return yc[:(wf_size+pad_size_2)]

def cali_zWave_predistort(cfg,z_dict):
    '''使用predistort函数修正Z脉冲'''
    All_Qubits = cfg.get('Chip').get('All_Qubits',[])
    wdLen=cfg.get('ADC_Meas').get('global').get('timeSeq').get('wdLen')
    probT0=cfg.get('ADC_Meas').get('global').get('timeSeq').get('probT0')
    gapTime=cfg.get('ADC_Meas').get('global').get('timeSeq').get('gapTime')

    new_z_dict={}
    for qubit in All_Qubits:
        zCali_dict=cfg.get(qubit).get('cali').get('zCali')
        # zCali = tuple(zCali_dict.get(key,0) for key in ['tau1', 'A1', 'tau2', 'A2', 'tau3', 'A3', 'tau4', 'A4','tau5','A5'])
        zCali_size = int((len(zCali_dict)+1)/2)
        zCali_tau = tuple(zCali_dict.get(key,0) for key in (f'tau{i}' for i in range(1,zCali_size+1)))
        zCali_A = tuple(zCali_dict.get(key,0) for key in (f'A{i}' for i in range(1,zCali_size+1)))
        zCali = tuple(zip(zCali_tau,zCali_A))
        z_kernal = z_dict[qubit]
        delay=probT0-gapTime-z_kernal.len #正常情况下的延迟时间
        
        if any(zCali_tau) and any(zCali_A):
            z_data_cali = predistort(tuple(z_kernal.data),z_kernal.sRate,zCali)  # 处理后Z增加了拖尾，即长度增加
            z_kernal = Wavedata(z_data_cali,z_kernal.sRate)
        #### 注意，这里对Z进行了拉长平移的处理
        z_wd = z_kernal.setLen(wdLen) >> delay
        new_z_dict[qubit] = z_wd
    return new_z_dict

def cali_Wave_delay(cfg,xy_dict,z_dict,mask_dict):
    '''对 xyWave/zWave/mask 做延迟校准'''
    All_Qubits = cfg.get('Chip').get('All_Qubits',[])
    new_xy_dict,new_z_dict,new_mask_dict={},{},{}
    for qubit in All_Qubits:
        Delay_dict = cfg.get(qubit).get('cali').get('Delay')
        xyDelay=Delay_dict.get('XY')
        zDelay=Delay_dict.get('Z')

        new_xy_dict[qubit] = xy_dict[qubit] >> xyDelay
        new_z_dict[qubit] = z_dict[qubit] >> zDelay
        new_mask_dict[qubit] = mask_dict[qubit] >> xyDelay
    return new_xy_dict,new_z_dict,new_mask_dict

def editWave(cfg,gateMap,gateindex=None):
    '''汇集波形处理过程'''
    #############################################################
    # 汇集波形处理过程
    xy_dict,z_dict = RawWave(cfg,gateMap,gateindex)
    xy_dict,mask_dict = cali_xyWave_carry(cfg,xy_dict)
    z_dict = cali_zWave_crossTalk(cfg,z_dict)
    z_dict = cali_zWave_predistort(cfg,z_dict)
    xy_dict,z_dict,mask_dict = cali_Wave_delay(cfg,xy_dict,z_dict,mask_dict)
    return xy_dict,z_dict,mask_dict

def hash_params(cfg,gateMap):
    '''哈希与波形编辑相关的参数'''
    All_Qubits = cfg.get('Chip').get('All_Qubits',[])
    ## 字典
    timeSeq = str(cfg.get('ADC_Meas').get('global').get('timeSeq'))
    gatemap = str(gateMap)
    # np.array
    matrix = str(cfg.get('Chip').get('crossTalkMatrix_pulse'))

    para_for_hash = [timeSeq,gatemap,matrix,]
    
    for qubit in All_Qubits:
        ## string or num
        carrywave_freq=cfg.get(qubit).get('para').get('drivCarryFreq')
        driv_scale_Q=cfg.get(qubit).get('cali').get('drivIQcali').get('scale_Q')
        driv_phase_Q=cfg.get(qubit).get('cali').get('drivIQcali').get('phase_Q')
        DelayXY = cfg[qubit]['cali']['Delay']['XY']
        DelayZ = cfg[qubit]['cali']['Delay']['Z']
        # dict
        gate = str(cfg[qubit]['gate'])
        zCali=str(cfg[qubit]['cali']['zCali'])

        para_for_hash.extend([qubit,carrywave_freq,driv_scale_Q,driv_phase_Q,
            DelayXY,DelayZ,gate,zCali])
    return hash(tuple(para_for_hash))

class GateEditor():
    cache = Cache(maxsize=256)
    
    def __init__(self,config):
        self.gateindex = GateIndex(config)
    
    def editWave(self,config,gateMap):
        cache_key = hash_params(config,gateMap)
        wave_tuple = self.cache.get(cache_key)

        if wave_tuple is None:
            wave_tuple = editWave(config,gateMap,gateindex=self.gateindex)
            self.cache.set(cache_key,wave_tuple)
        else:
            log.info(f'GateEditor.editWave use cache {cache_key}')
        # wave_tuple_copy = copy.deepcopy(wave_tuple) # copy 耗时10ms
        return wave_tuple
