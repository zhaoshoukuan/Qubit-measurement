#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   savefile.py
@Time    :   2019/04/11 14:51:55
@Author  :   skzhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, zhaogroup-lala
@Desc    :   None
'''


import numpy as np 
from datetime import datetime
import os
import time
from pathlib import Path


def SaveFile(tags1,tags2,*value):

        #self.now = time.strftime("%H %M %S", time.localtime()) + '.txt'
    now = ''.join((tags2,'_',time.strftime("%H%M%S", time.localtime()),'.npz'))
    l = ''.join((tags2 , '.lock'))
    t = time.strftime('%y%m%d') 
    p = Path('d:/skzhao/Data') / tags1
    if os.path.exists(p) == 0:
        os.makedirs(p)
    
    #with open(p / l,'w') as f:
        #f.write('1')
 
    pa = p / now  
        
    #for i in range(10):
    try:
                #with portalocker.Lock(pa,'wb') as f:
        with open(pa,'wb') as f:
            np.savez(f,row=value[0],col=value[1],s=value[2])
    except PermissionError:
        pass
    #os.remove(p / l)
    return ''.join(('d:/skzhao/Data','/','/',tags1,'/',tags2))