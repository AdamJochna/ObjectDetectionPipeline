import pandas as pd
import numpy as np

pF = pd.read_csv('/home/adam/Desktop/kaggleOID/EVALcsvs/predictionsFRCNN.csv')
pY = pd.read_csv('/home/adam/Desktop/kaggleOID/EVALcsvs/predictionsYOLOv3_OID.csv')

print(pF["Score"].describe())
print(50*'#')
print(pY["Score"].describe())

def make_score_uniform(df,low,high):

    df=df.sort_values(by=['Score'],ascending=False)
    df['orig_idx']=df.index
    df=df.reset_index(drop=True)
    df['Score']=(len(df.index) + (-1)*df.index)/len(df.index)
    df['Score']=df['Score']*(high-low) + low

    df=df.set_index('orig_idx')
    df=df.sort_index()
    del df.index.name

    return df

def merge_and_out(df1,df2):
    return pd.concat([df1,df2])


def makeoverlap(pF,pY,l,h):
    pF=make_score_uniform(pF,low=0,high=1)
    pY = make_score_uniform(pY, low=l, high=h)

    return merge_and_out(pF,pY)

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.7):
    if len(boxes) == 0:
        return boxes, probs

    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    if np.all(np.less(x1, x2)) != True:
        print("err")
        return boxes, probs

    if np.all(np.less(y1, y2)) != True:
        print("err")
        return boxes, probs

    boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        area_int = np.maximum(0, xx2_int - xx1_int) * np.maximum(0, yy2_int - yy1_int)
        overlap = area_int/(area[i] + area[idxs[:last]] - area_int + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))

    boxes = boxes[pick].astype("float")
    probs = probs[pick]
    return boxes, probs

l=0.67
h=0.96

dfMerg=makeoverlap(pF.copy(),pY.copy(),l,h)
dfMerg=dfMerg.sort_values(by=['ImageID'],ascending=True)
dfMerg=dfMerg.reset_index(drop=True)

grouped = dfMerg.groupby('ImageID')
dflist=[]
idx=0

for name, group in grouped:
    idx+=1
    print(idx)
    grouped2 = group.groupby('LabelName')

    for name2, group2 in grouped2:

        boxes=group2[['YMin','XMin','YMax','XMax']].values
        probs=group2['Score'].values
        boxes, probs = non_max_suppression_fast(boxes, probs, overlap_thresh=0.5)

        probs=np.expand_dims(probs,axis=1)
        namearr = np.expand_dims(np.array([name for i in range(boxes.shape[0])]), axis=1)
        name2arr = np.expand_dims(np.array([name2 for i in range(boxes.shape[0])]), axis=1)
        all=np.concatenate([namearr,name2arr,probs,boxes], axis=1)

        alldf=pd.DataFrame(all,columns=dfMerg.columns)
        alldf=alldf.loc[alldf['Score'].astype(np.float) > 0.5]
        dflist.append(alldf.copy())

dfFinal=pd.concat(dflist)
print(dfFinal)

dfFinal.to_csv("/home/adam/Desktop/kaggleOID/EXPcsvs/evalafterNMS05.csv", index=False)

