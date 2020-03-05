import numpy as np 
import visa, time
from call import savefile as sf
import matplotlib.pyplot as plt

rm = visa.ResourceManager()
dpo = rm.open_resource('TCPIP::10.122.7.190')

ch, line = 4, 'D3' 

def get_Trace(ch=3, start=1):
    #dpo.write('ACQUIRE:STOPAFTER SEQUENCE') #相当于single按钮
    dpo.write(':DAT:SOU CH%d' % ch)
    dpo.write(':WFMOutpre:ENCdg ASCii') #编码模式
    dpo.write(':DAT:START %d' % start)
    #stop = dpo.query_ascii_values('WFMOUTPRE:NR_PT?') #record length
    stop = dpo.query_ascii_values('HORIZONTAL:RECORDLENGTH?')
    xstep = dpo.query_ascii_values(':WFMI:XIN?')
    print(stop)
    dpo.write(':DAT:STOP %g' % stop[0])
    data = np.array(dpo.query_ascii_values('CURV?'))
    y_zero = np.array(dpo.query_ascii_values(':WFMO:YZER?' ))
    y_multi = np.array(dpo.query_ascii_values(':WFMOutpre:YMUlt?' ))
    y_units = dpo.query('WFMOutpre:YUNit? ').strip('\n').strip('""')
    y = y_zero + y_multi * data
    
    x_zero = np.array(dpo.query_ascii_values(':WFMO:XZER?'))
    x_incr = np.array(dpo.query_ascii_values(':WFMOutpre:XINcr?'))
    x_units = dpo.query('WFMOutpre:XUNit? ').strip('\n').strip('""')
    x = [x_zero+x_incr*i for i in range(len(data))]

    return np.array(x), y, x_units, y_units

def read():

    i, i_max= 0, 60/6
    while True:
        
        t = t = time.strftime('%Y%m%d_%H %M %S',time.localtime(time.time()))
        x, y, x_units, y_units = get_Trace(ch, start=1)
        x, y = x*1e6, y*1e3
        data = x, y, None
        p = sf.SaveFile('DPO',''.join((line,'_','ch%s'%str(ch))),*data)

        plt.figure(figsize=(7,6))
        plt.plot(x,y)
        plt.title(r'$ch = %g,line=%s $' %(ch,line))
        plt.xlabel('time /us')
        plt.ylabel('voltage /mV')
        plt.savefig(r'C:\Users\23967\Desktop\figure\dpo\%s.png'%''.join(('ch%s'%str(ch),'_',t)))
        plt.pause(600)
        plt.close()
        
        if i==0:
            with open(r'D:\skzhao\file_name\file_name.txt', mode='a') as filename:
                filename.write(p)
                filename.write('\n')
        if i==int(i_max):
            break
        i = i+1

if __name__ == '__main__':
    read()
