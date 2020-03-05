import awg_setup as asp
import numpy as np
import visa

rm = visa.ResourceManager()
awg = rm.open_resource('TCPIP::10.122.7.133')

#定义sequence
def Sequence():
    ASP = asp.AWG_SETUP(awg,'Ramsey',1,choice='seq')
    h=[[1]*401,[1]*401]
    t=[[89900]*401,89900-86/2-np.linspace(1,20001,401)]
    w=[[43]*401,[43]*401]
    ASP.sequence(t_end=t,width=w,height=h,tag='+-')
    
#定义台阶
def Square_wave():
    ASP = asp.AWG_SETUP(awg,'z_test',5)
    h = [0]
    t = [20000]
    w = [10000]
    ASP.square_wave(t,w,h,tag='+-')
    
#定义cos包络的正弦函数
def Sin_wave():
    ASP = asp.AWG_SETUP(awg,'trigger_lo',3)
    f = [50e6]
    phi = [0]
    h = [1]
    t = [18000]
    w = [2000]
    ASP.sin_envelope(f,phi,t,w,h)

def drag_wave():
    ASP = asp.AWG_SETUP(awg,'Ex_Q',7)
    f = [50e6]
    phi = [0]
    h = [1]
    t = [89700]
    w = [32]
    ASP.drag_envelope(f,phi,t,w,h)
    
    
#定义cos包络的余弦函数
def Cos_wave():
    ASP = asp.AWG_SETUP(awg,'Readout_Q',)
    f = [50e6]
    phi = [0]
    h = [1]
    t = [91000]
    w = [1000]
    ASP.sin_envelope(f,phi,t,w,h)

#定义方波包络
def Square_Envelope():
    ASP = asp.AWG_SETUP(awg,'Ex_I',3)
    f = [50e6]
    phi =[np.pi/2]
    h = [1]
    t = [91000]
    w = [1000]
    ASP.square_envelope(f,phi,t,w,h)
    
#add marker to waveform
def AddMarkToWaveform():
    ASP = asp.AWG_SETUP(awg,'Readout_Q',4)
    h1 = [1]
    t1 = [91000]
    w1 = [1000]
    h2 = [1]
    t2 = [90994]
    w2 = [1000]
    ASP.add_marker_to_waveform(mk2=(t2,w2,h2),mk1=(t1,w1,h1),mk3=None,mk4=None,start=0,size=None,name=None,version='AWG5014C')

Square_wave()
#Sin_wave()
#Cos_wave()
#Square_Envelope()
#Sequence()
#AddMarkToWaveform()
#drag_wave()















