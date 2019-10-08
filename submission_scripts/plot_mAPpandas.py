import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)

mAPs_path='/home/adam/Desktop/kaggleOID/mAPs.txt'
descriptors_path='/home/adam/Desktop/kaggleOID/data/descriptors.csv'
input_path='/home/adam/Desktop/kaggleOID/data/annotsfull.csv'
input_path_eval='/home/adam/Desktop/kaggleOID/eval/challenge-2019-validation-detection-bbox_expanded_v2.csv'
maxlimit=-1

classes_total = {}
classes_total_eval = {}
eng_mapping = {}
df=pd.DataFrame()

def mapdict(dict,x):
    return dict[str(x)]

with open(descriptors_path, 'r') as f:
    for i, line in enumerate(f):
        line_split = line.strip().split(',')
        (coded, engcoded) = line_split
        eng_mapping[coded] = engcoded
        classes_total[coded] = 0
        classes_total_eval[coded] = 0

with open(input_path, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        if i == maxlimit:
            break

        line_split = line.strip().split(',')
        if len(line_split) == 13:
            (_, _, class_name, _, _, _, _, _, _, _, _, _, _) = line_split
        if len(line_split) == 7:
            (_, class_name, _, _, _, _, _) = line_split

        if class_name in classes_total:
            classes_total[class_name] += 1

with open(input_path_eval, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        if i == maxlimit:
            break

        line_split = line.strip().split(',')
        if len(line_split) == 13:
            (_, _, class_name, _, _, _, _, _, _, _, _, _, _) = line_split
        if len(line_split) == 7:
            (_, class_name, _, _, _, _, _) = line_split

        if class_name in classes_total_eval:
            classes_total_eval[class_name] += 1

models_names=[]

with open(mAPs_path,'r') as f:
    while True:
        line1 = f.readline()
        line2 = f.readline()

        if line1:
            name = line1.split('/')[-1].split('.')[0]
            dict = eval(line2.replace("nan", "0"))
            dict = {k[61:]:v for (k,v) in dict.items()}
            tmp_df=pd.DataFrame.from_records([dict]).T
            tmp_df.set_axis([name], axis=1)
            df=pd.concat([df,tmp_df], axis=1)
            models_names.append(name)
        else:
            break

df.drop(df.index[0],inplace=True)
df.reset_index(inplace=True)
df.rename(columns={"index": "coded"},inplace=True)

df['maxAPall'] = df[models_names].max(axis=1)
df['tr_count'] = df['coded'].apply(lambda x : mapdict(classes_total,x))
df['eval_count'] = df['coded'].apply(lambda x : mapdict(classes_total_eval,x))
df['maxModel'] = df[models_names].idxmax(axis=1)

df['eng_map'] = df['coded'].apply(lambda x : mapdict(eng_mapping,x))
df['eng_map'] = df['eng_map'].str.lower()

print(df.sort_values(by=['eval_count']))

#print(list(df.loc[df['eng_map'].str.contains("human")]['coded'].values))

model_for_class={}
model_for_modelname={'evYOLOv3':'/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsYOLOv3_OID.csv',
                     'evFRCNN':'/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsFRCNN.csv',
                     'evInception':'/home/adam/Desktop/kaggleOID/FULLcsvs/predictionsInception.csv',
                     'evalFRCNN150cls':'/home/adam/Desktop/kaggleOID/FULLcsvs/fullFrcnn150cls.csv',
                     'evalZFturbo152':'/home/adam/Desktop/kaggleOID/FULLcsvs/fullZFturbo152.csv',
                     'evalZFturbo101':'/home/adam/Desktop/kaggleOID/FULLcsvs/fullZFturbo101.csv',
                     'evalZFturboMerg':'/home/adam/Desktop/kaggleOID/FULLcsvs/fullZFturboMerged.csv',
                     'evalClique_0_5_1600':'/home/adam/Desktop/kaggleOID/FULLcsvs/fullClique_05_1600_csvf_28_01h_08m.csv'}

for index, row in df.iterrows():
    if row['eval_count']<20:
        model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
    elif row['eval_count']<100:
        if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.03:
            model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
        else:
            model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
    else:
        if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.01:
            model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
        else:
            model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']

print(model_for_class)


# setting num1
# for index, row in df.iterrows():
#     if row['eval_count']<50:
#         model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     elif row['eval_count']<100:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.03:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     else:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.01:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']

# setting num2
# for index, row in df.iterrows():
#     if row['eval_count']<50:
#         model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     elif row['eval_count']<100:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.05:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     else:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.02:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']

# setting num3
# for index, row in df.iterrows():
#     if row['eval_count']<20:
#         model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     elif row['eval_count']<100:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.03:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']
#     else:
#         if row['maxAPall'] > row['evalClique_0_5_1600'] + 0.01:
#             model_for_class[row['coded']] = model_for_modelname[row['maxModel']]
#         else:
#             model_for_class[row['coded']] = model_for_modelname['evalClique_0_5_1600']