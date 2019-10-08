import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


listYOLO=[]
listFRCNN=[]
listINCEPT=[]

currimid=''
cnt=0

def Frcnnfunc(x):
    return x/(2-x)

def Yolofunc(x):
    return np.power(x,0.7)

def Incepfunc(x):
    return x

def postFrcnnfunc(x):
    return np.power(x,2)

def postYolofunc(x):
    return x

def postIncepfunc(x):
    return x


with open("/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsYOLOv3_OID.csv",'r') as f:
    line = f.readline()
    while True:
        line = f.readline()
        ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
        Score=Yolofunc(float(Score))
        if float(Score)>0.05:
            listYOLO.append(postYolofunc(Score))
        if ImageID!=currimid:
            currimid=ImageID
            cnt+=1
            if cnt==1000:
                cnt = 0
                break

with open("/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsFRCNN.csv",'r') as f:
    line = f.readline()
    while True:
        line = f.readline()
        ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
        Score=Frcnnfunc(float(Score))
        if float(Score) > 0.05:
            listFRCNN.append(postFrcnnfunc(Score))
        if ImageID != currimid:
            currimid = ImageID
            cnt += 1
            if cnt == 1000:
                cnt = 0
                break

with open("/home/adam/Desktop/kaggleOID/inception/predictionsInception.csv",'r') as f:
    line = f.readline()
    while True:
        line = f.readline()
        ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
        Score=Incepfunc(float(Score))
        if float(Score) > 0.05:
            listINCEPT.append(postIncepfunc(Score))
        if ImageID != currimid:
            currimid = ImageID
            cnt += 1
            if cnt == 1000:
                cnt = 0
                break


bins = np.linspace(0, 1, 50)

plt.hist([np.array(listYOLO),np.array(listFRCNN),np.array(listINCEPT)],bins, label=['listYOLO', 'listFRCNN', 'listINCEPT'])
plt.legend(loc='upper right')
plt.ylim(0,5000)

plt.savefig('/home/adam/Desktop/kaggleOID/sampleplt.png')