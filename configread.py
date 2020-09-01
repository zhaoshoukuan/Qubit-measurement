import pymongo, time, numpy as np, datetime
from pathlib import Path
import inspect
from configparser import ConfigParser
from mongoengine import (BooleanField, ComplexDateTimeField, Document,
                         FileField, IntField, ListField, ReferenceField,
                         StringField, ObjectIdField)

PATH = r'D:\QuLabData\config\status.ini'
now = datetime.datetime.now()

class Record(Document):
    title = StringField(max_length=100)
    comment = StringField()
    created_time = ComplexDateTimeField(default=now)
    finished_time = ComplexDateTimeField(default=now)
    modified_time = ComplexDateTimeField(default=now)
    hidden = BooleanField(default=False)
    children = ListField(ReferenceField('Record'))
    tags = ListField(StringField(max_length=50))
    loc = StringField()
    imagefield = FileField(collection_name='images')
    work = ReferenceField('CodeSnippet')
    notebook = ReferenceField('Notebook')
    notebook_index = IntField(min_value=0)

def loadconfig(path=None):
    path == path if path else PATH
    config = ConfigParser()
    return config.read(path)

def saveconfig(config,name,tag=[],basepath=None):
    fname = f"{name}_{time.strftime('%Y%m%d%H%M%S')}.ini"
    if basepath:
        path = basepath / fname
    else:
        pathnow = Path('D:\QuLabData\config') / time.strftime('%Y') / time.strftime('%m%d')
        pathnow.mkdir(parents=True, exist_ok=True)
        path = pathnow/fname
    # record = Record(title=''.join(('config',name)),tags=tag,loc=str(path))
    # record.children.append(record)
    # record.save(signal_kwargs=dict(finished=True))

    with open(path,'w') as file:
        config.write(file)
    return path

def updatefileattr(module,name='default',tag=[]):
    cp = ConfigParser()
    functions_list = [o for o in inspect.getmembers(module) if inspect.isfunction(o[1])]
    for i in functions_list:
        insp = inspect.getfullargspec(i[1])
        paras, defaults = insp[0][::-1], insp[3]
        cp.add_section(i[0])
        for j,k in enumerate(paras):
            if defaults and j < len(defaults):
                cp.set(i[0],k,str(defaults[-j-1]))
            else:
                cp.set(i[0],k,'None')
    path = saveconfig(cp,name,tag=tag)
    return path
