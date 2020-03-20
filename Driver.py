import logging
import time
from collections import deque
import visa
import numpy as np
from itertools import count
from scipy import fftpack, signal
import qulab.optimize as op
from qulab import BaseDriver

from .AlazarTechWrapper import (AlazarTechDigitizer, AutoDMA, DMABufferArray,
                                configure, initialize)
from .exception import AlazarTechError

log = logging.getLogger(__name__)

rm = visa.ResourceManager()
#awg133 = rm.open_resource('TCPIP::10.122.7.133')
#awg132 = rm.open_resource('TCPIP::10.122.7.132')
AFG = rm.open_resource('GPIB::30')
#awg100 = rm.open_resource('TCPIP::10.122.7.100')

def getSamplesPerRecode(numOfPoints):
    samplesPerRecord = (numOfPoints // 64) * 64
    if samplesPerRecord < numOfPoints:
        samplesPerRecord += 64
    return samplesPerRecord - 128


def getExpArray(f_list, numOfPoints, weight=None, sampleRate=1e9):
    e = []
    t = np.arange(0, numOfPoints, 1) / sampleRate
    if weight is None:
        weight = np.ones(numOfPoints)
    for f in f_list:
        e.append(weight * np.exp(-1j * 2 * np.pi * f * t))
    return np.asarray(e).T


class Driver(BaseDriver):
    def __init__(self, systemID=1, boardID=1, config=None, **kw):
        super().__init__(**kw)
        self.dig = AlazarTechDigitizer(systemID, boardID)
        self.config = dict(n=1024,
                           sampleRate=1e9,
                           f_list=[50e6],
                           weight=None,
                           awg=None,
                           repeats=512,
                           maxlen=512,
                           ARange=1.0,
                           BRange=1.0,
                           trigLevel=0.0,
                           triggerDelay=0,
                           triggerTimeout=0,
                           recordsPerBuffer=64,
                           bufferCount=512)
        self.config['samplesPerRecord'] = getSamplesPerRecode(self.config['n'])
        self.config['e'] = getExpArray(self.config['f_list'], self.config['samplesPerRecord'],
                                       self.config['weight'],
                                       self.config['sampleRate'])
        if config is not None:
            self.set(**config)
        initialize(self.dig)
        configure(self.dig, **self.config)

    def set(self, **cmd):
        if 'n' in cmd:
            cmd['samplesPerRecord'] = getSamplesPerRecode(cmd['n'])
            cmd['n'] = getSamplesPerRecode(cmd['n'])
        self.config.update(cmd)

        if self.config['repeats'] % self.config['recordsPerBuffer'] != 0:
            self.config['repeats'] = (
                self.config['repeats'] // self.config['recordsPerBuffer'] +
                1) * self.config['recordsPerBuffer']

        if any(key in ['f_list', 'n', 'weight', 'sampleRate'] for key in cmd):
            self.config['e'] = getExpArray(self.config['f_list'],
                                           self.config['n'],
                                           self.config['weight'],
                                           self.config['sampleRate'])

        if any(key in [
                'ARange', 'BRange', 'trigLevel', 'triggerDelay',
                'triggerTimeout'
        ] for key in cmd):
            configure(self.dig, **self.config)

    def setValue(self, name, value):
        self.set(**{name: value})

    def getValue(self, name):
        return self.config.get(name, None)

    def _aquireData(self, samplesPerRecord, repeats, buffers, recordsPerBuffer,
                    timeout):
        with AutoDMA(self.dig,
                     samplesPerRecord,
                     repeats=repeats,
                     buffers=buffers,
                     recordsPerBuffer=recordsPerBuffer,
                     timeout=timeout) as h:

            if self.config['awg']:
                #awg100.write('TRIG')
                #awg100.write('AWGC:RUN')
                AFG.write('*TRG')
                #AFG.query('*OPC?')
                
            yield from h.read()

    def getData(self,offset):
        samplesPerRecord = self.config['samplesPerRecord']
        recordsPerBuffer = self.config['recordsPerBuffer']
        repeats = self.config['repeats']
        e = self.config['e']
        n = e.shape[0]
        maxlen = self.config['maxlen']
     
        A, B = [], []
        retry = 0
        while retry < 3:
            try:
                for index , (chA, chB) in zip(count(), self._aquireData(
                        samplesPerRecord,
                        repeats=repeats,
                        buffers=None,
                        recordsPerBuffer=recordsPerBuffer,
                        timeout=1)):
                    A_lst = chA.reshape((recordsPerBuffer, samplesPerRecord))
                    B_lst = chB.reshape((recordsPerBuffer, samplesPerRecord))
                    A.append(A_lst)
                    B.append(B_lst)
                
                    if repeats == 0 and index*recordsPerBuffer >= maxlen:
                        break
                A = np.asarray(A)
                B = np.asarray(B)
                
                A = A.flatten().reshape(A.shape[0]*A.shape[1], A.shape[2])
                B = B.flatten().reshape(B.shape[0]*B.shape[1], B.shape[2])
                if offset:
                    A = (A.T-np.mean(A,axis=1)).T
                    B = (B.T-np.mean(B,axis=1)).T
                return A, B

            except AlazarTechError as err:
                log.exception(err.msg)
                if err.code == 518:
                    raise SystemExit(2)
                else:
                    pass
            time.sleep(0.1)
            retry += 1
        else:
            raise SystemExit(1)
    
    def demodulation(self,fft=False, avg=False, hilbert=False, offset=True, is2ch=True):
        e = self.config['e']
        n = e.shape[0]
        A_lst, B_lst = self.getData(offset)
        if hilbert:
            Analysis_cos = signal.hilbert(A_lst,axis=1)
            Analysis_sin = signal.hilbert(B_lst,axis=1)
            # theta = np.angle(Analysis_cos) - np.angle(Analysis_sin)
            # Analysis_sin *= np.exp(1j*(theta))
            # A_lst, B_lst = (np.real(Analysis_cos) + np.real(Analysis_sin)), (np.imag(Analysis_cos) + np.imag(Analysis_sin)) 
            if is2ch:
                A_lst, B_lst = (np.real(Analysis_cos) - np.imag(Analysis_sin)), (np.imag(Analysis_cos) + np.real(Analysis_sin))
            else: 
                A_lst, B_lst = np.real(Analysis_cos), np.imag(Analysis_cos)
        if fft:
            A_lst = (A_lst[:, :n]).dot(e)
            B_lst = (B_lst[:, :n]).dot(e)
        if avg:
            return A_lst.mean(axis=0), B_lst.mean(axis=0)
        else:
            return A_lst, B_lst

    def getIQ(self,fft=True, avg=False, hilbert=True, offset=True, is2ch=True):
        return self.demodulation(fft, avg, hilbert, offset, is2ch)

    def getTraces(self,fft=False, avg=True, hilbert=True, offset=True, is2ch=True):
        return self.demodulation(fft, avg, hilbert, offset, is2ch)
