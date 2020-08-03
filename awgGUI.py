import tkinter as tk
import visa, asyncio

def ch_skew(awg):
    
    awggui = tk.Tk()
    awggui.title('awg-async')
    awggui.geometry('500x300')
    labellist = ['ch1','ch2','ch3','ch4']

    ch_time = []
    for j,i in enumerate(labellist):
        label = tk.Label(awggui,text=i,bg='green',font=('Arial', 12),width=10,height=2)
        label.place(x=3,y=75*j)
        timevar = tk.StringVar(awggui)
    time = tk.Entry(awggui, textvariable=timevar, font=('Arial', 12))
    time.place(x=120,y=75*j+10)
    ch_time.append(timevar)
    
    def hit_me():
        global ch_time
        for i in ch_time:
            t = eval(i.get())
            print(t)
    tk.Button(awggui,text='Enter', font=('Arial', 12), width=15, height=3, command=hit_me).place(x=340,y=110)
    awggui.mainloop()