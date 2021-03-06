import visa, numpy as np, visa, time
import waveform_new as wn
import tkinter as tk

t_lst = np.linspace(-50000,50000,100000)*1e-9

rm = visa.ResourceManager()
awg = rm.open_resource('TCPIP::10.122.7.100')

def update_marker(awg,
                    name,
                    mk1=None,
                    mk2=None,
                    mk3=None,
                    mk4=None,
                    start=0,
                    size=None):
    def format_marker_data(markers, bits):
        values = 0
        for i, v in enumerate(markers):
            v = 0 if v is None else np.asarray(v, dtype=int)
            values += v << bits[i]
        return values

    values = format_marker_data([mk1, mk2], [6, 7])
    if size is None:
        message = 'WLIST:WAVEFORM:MARKER:DATA "%s",%d,' % (name, start)
    else:
        message = 'WLIST:WAVEFORM:MARKER:DATA "%s",%d,%d,' % (name, start,
                                                                size)
    awg.write_binary_values(message,
                                values,
                                datatype=u'B',
                                is_big_endian=False,
                                termination=None,
                                encoding=None)
defaulttime = np.loadtxt(r'D:\QuLabData\awgGui\state.txt',delimiter=',',dtype=list)


def ch_skew():
    
    awggui = tk.Tk()
    awggui.title('awg-async')
    awggui.geometry('500x300')
    labellist = ['ch1','ch2','ch3','ch4']
    # await cw.genwaveform(awg,labellist,[1,2,3,4],t_list=t_lst)

    ch_time = []
    for j,i in enumerate(labellist):
        label = tk.Label(awggui,text=i,bg='green',font=('Arial', 12),width=10,height=2)
        label.place(x=3,y=75*j)
        timevar = tk.StringVar(awggui,str(defaulttime[j]).strip('"]","["'))
        time = tk.Entry(awggui, textvariable=timevar, font=('Arial', 12))
        time.place(x=120,y=75*j+10)
        ch_time.append(timevar)
    
    def hit_me():
        # global ch_time
        wav = []
        tlst = []
        for i in ch_time:
            t = eval(i.get())
            tlst.append(t)
            pulse = wn.square(5000/1e9) << (2500+t)/1e9
            wav.append(pulse(t_lst))

        for i, j in enumerate(['ch1','ch2','ch3','ch4']):
            update_marker(awg,j,wav[i]) 
        if isinstance(tlst[0],int):
            with open(r'D:\QuLabData\awgGui\state.txt', mode='w') as filename:
                filename.write(str(tlst))
                filename.write('\n')         
        print('Done!')
    tk.Button(awggui,text='Enter', font=('Arial', 12), width=15, height=3, command=hit_me).place(x=340,y=110)
    awggui.mainloop()

ch_skew()