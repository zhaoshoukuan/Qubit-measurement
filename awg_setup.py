#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   awg_setup.py
@Time    :   2019/06/02 22:37:43
@Author  :   skzhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, zhaogroup-lala
@Desc    :   None
'''

# here put the import lib

import numpy as np 

t = np.linspace(0,100000,250000)
delta = 50e6

class AWG_SETUP():
    
    def __init__(self,awg,name1,ch,choice='wave'):
        self.name = name1
        self.awg = awg
        self.ch = ch
   
        name_inawg = self.awg.query('WLIS:LIST?').strip('\n' '\"').split(',')
        if choice is 'wave':
            if name1 not in name_inawg:
                self.awg.write('WLIS:WAV:NEW "%s",%d,%s;' %(name1,len(t),'REAL'))
        self.awg.write('SOURCE%d:WAVEFORM "%s"' % (self.ch, self.name))
    
    def step(self,t_end,width):
        return (t>(t_end-width))*1-(t>t_end)*1
    
    def updatewaveform(self,name, points):
        message = 'WLIST:WAVEFORM:DATA "%s",%d,' % (name, 0)
        message = message + ('%d,' % len(points))
        values = points.clip(-1,1)
        self.awg.write_binary_values(message, values, datatype=u'f',is_big_endian=False,termination=None, encoding=None)
    
    def wave(self,t_end=0,width=0,height=0,tag=None):
        pulse_want = 0
        for i in range(len(t_end)):
            if tag=='+-':
                pulse_want += (self.step(t_end[i],width[i]))*height[i]
                pulse_want = pulse_want*2-1
            else:
                pulse_want = pulse_want + self.step(t_end[i],width[i])*height[i]
        return pulse_want

#waveform
    def square_wave(self,t_end=0,width=0,height=0,tag=None,name=None):
        if name is None:
            name = self.name
        pulse_want = self.wave(t_end,width,height,tag)
        self.updatewaveform(name,pulse_want)
        return pulse_want
        
    def sin_envelope(self,f=0,phi=0,t_end=0,width=0,height=0,tag=None,name=None):
        if name is None:
            name = self.name
        #pulse_want = self.wave(t_end,width,height,tag)*np.sin(np.pi/width[0]*(t-t_end[0]))*np.sin(2*np.pi*f[0]*1e-9*t+phi)
        pulse_want = self.wave(t_end,width,height)*1/2*(np.cos(2*np.pi/width[0]*(t-t_end[0]+width[0]/2))+1)*np.sin(2*np.pi*f[0]*1e-9*t+phi)
        self.updatewaveform(name,pulse_want)
        return pulse_want

    def drag_envelope(self,f=0,phi=0,t_end=0,width=0,height=0,tag=None,name=None):
        if name is None:
            name = self.name
        #pulse_want = self.wave(t_end,width,height,tag)*np.sin(np.pi/width[0]*(t-t_end[0]))*np.sin(2*np.pi*f[0]*1e-9*t+phi)
        pulse_want = self.wave(t_end,width,height)*1/2*(np.cos(2*np.pi/width[0]*(t-t_end[0]+width[0]/2))+1-0.5*1e9*np.sin(2*np.pi/width[0]*(t-t_end[0]+width[0]/2))/width[0]/252e6)*np.sin(2*np.pi*f[0]*1e-9*t+phi)
        self.updatewaveform(name,pulse_want)
        return pulse_want
    
    def square_envelope(self,f=0,phi=0,t_end=0,width=0,height=0,tag=None,name=None):
        if name is None:
            name = self.name
        pulse_first = 1
        for i in range(len(f)):
            pulse_want = pulse_first*np.sin(2*np.pi*f[i]*1e-9*t+phi[i]) 
        pulse_want = pulse_want*self.wave(t_end,width,height,tag)
        self.updatewaveform(name,pulse_want)
        return pulse_want
#sequence 
    def sequence(self,t_end,width,height,tag=None,name=None):
        if name is None:
            name = self.name
        def set_sequence_wave(i,t_end,width,height, name, wait='OFF', goto='NEXT', repeat='INFinite', jump0='ATRigger', jump1='NEXT'):
            
            name_inawg = self.awg.query('WLIS:LIST?').strip('\n' '\"').split(',')
            if ''.join((name,'_','seq','%d')) %i not in name_inawg:
                self.awg.write('WLIS:WAV:NEW "%s",%d,%s;' % (''.join((name,'_','seq','%d')) %i, len(t), 'REAL'))
            
            self.updatewaveform(''.join((name,'_','seq','%d')) %i, self.wave(t_end,width,height,tag))
            
            self.awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' %(i, 1, name, ''.join((name,'_','seq','%d')) %i))
            self.awg.write('SLIS:SEQ:STEP%d:WINP "%s", %s' % (i, name, wait))
            self.awg.write('SLIS:SEQ:STEP%d:GOTO "%s", %s' % (i, name, goto))
            self.awg.write('SLIS:SEQ:STEP%d:RCO "%s", %s' % (i, name, repeat))
            self.awg.write('SLIS:SEQ:STEP%d:EJIN "%s", %s' % (i, name, jump0))
            self.awg.write('SLIS:SEQ:STEP%d:EJUM "%s", %s' % (i, name, jump1))
            print(i)
            self.awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' % (i, 2, name, 'Readout_I'))
            self.awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' % (i, 3, name, 'Readout_Q'))
            self.awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' % (i, 4, name, 'trigger_zsk'))
            #self.awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' % (i, 5, name, 'Ex'))

        i = 1 #step 
        def get_sequence_name():
            ret = []
            slist_size = int(self.awg.query("SLIS:SIZE?"))
            for i in range(slist_size):
                ret.append(self.awg.query("SLIS:NAME? %d" % (i+1)).strip("\"\n '"))
            return ret
        if  name not in get_sequence_name():
            self.awg.write('SLIS:SEQ:NEW "%s", %d, %d' % (name, len(height[0]), 5))
        for j in range(len(height[0])):
            num = j
    
            x, y, z = [], [], []
            for k in range(len(height)):
                x.append(t_end[k][num])
                y.append(width[k][num])
                z.append(height[k][num])
            set_sequence_wave(i,x,y,z,name, wait='ATR', goto='NEXT', repeat='INF', jump0='BTR', jump1='NEXT')
            i += 1
        self.awg.write('SOUR%d:CASS:SEQ "%s", %d' % (self.ch, self.name, 1))

#add marker to waveform
    def add_marker_to_waveform(self,mk1=None,mk2=None,mk3=None,mk4=None,start=0,size=None,name=None,version='AWG5208'):
        if name is None:
            name = self.name
    
        if mk1 is not None:
            t_end, width, height = mk1
            mk1 = self.wave(t_end,width,height,tag=None)
        if mk2 is not None:
            t_end, width, height = mk2
            mk2 = self.wave(t_end,width,height,tag=None)
        if mk3 is not None:
            t_end, width, height = mk3
            mk3 = self.wave(t_end,width,height,tag=None)
        if mk4 is not None:
            t_end, width, height = mk4
            mk4 = self.wave(t_end,width,height,tag=None)
            
        def format_marker_data(markers,bits):
            values = 0
            for i, v in enumerate(markers):
                v = 0 if v is None else np.asarray(v)
                values += v << bits[i]
            return values
        
        if version is 'AWG5014C':
            values = format_marker_data([mk1, mk2], [6, 7])
        elif version is 'AWG5208':
            values = format_marker_data([mk1,mk2,mk3,mk4],[7,6,5,4])

        if size is None:
            message = 'WLIST:WAVEFORM:MARKER:DATA "%s",%d,' % (name,start)
        else:
            message = 'WLIST:WAVEFORM:MARKER:DATA "%s",%d,%d,' % (name,start,size)
        
        self.awg.write_binary_values(message,values,datatype=u'B',is_big_endian=False,termination=None,encoding=None)

#delete waveform or sequence    
    def DEL(self,name='ALL',tag='wave'):
        if tag=='seq':
            self.awg.write('SLIS:SEQ:DEL %s'%name)
        if tag=='wave':
            self.awg.write('wLIST:WAVEFORM:DELETE %s'% name)
