import numpy as np
import copy
import json

def get_data(input_path,descriptors_path,expectedcount,maxlimit,restricted_classes=None):
    classes_total = {}
    probabilities = {}

    with open(descriptors_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split(',')
            (coded, engcoded) = line_split
            classes_total[coded] = 0

    classes_total['bg']=0

    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i == maxlimit:
                break

            line_split = line.strip().split(',')
            if len(line_split)==13:
                (_, _, class_name, _, _, _, _, _, _, _, _, _, _) = line_split
            if len(line_split)==7:
                (_, class_name, _, _, _, _, _) = line_split

            if class_name in classes_total:
                classes_total[class_name] += 1

    for class_name in classes_total:
        if classes_total[class_name]!=0:
            if restricted_classes==None:
                probabilities[class_name] = min(1, expectedcount / classes_total[class_name])
            else:
                if class_name in restricted_classes:
                    probabilities[class_name] = min(1, expectedcount / classes_total[class_name])
                else:
                    probabilities[class_name] = 0
        else:
            probabilities[class_name]=0

    all_data=[]
    class_mapping = {}
    classes_count = {}
    eng_mapping = {}

    with open(descriptors_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split(',')
            (coded, engcoded) = line_split
            eng_mapping[coded] = engcoded
            class_mapping[coded] = i
            classes_count[coded] = 0

    eng_mapping['bg'] = 'bg'
    class_mapping['bg'] = len(class_mapping)
    classes_count['bg'] = 0

    tmpimg = {'filepath':None , 'ImageID':None ,'width':None ,'height':None ,'bboxes':[] ,'imageset':'train','maxProb':0}

    with open(input_path,'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i == maxlimit:
                break
            if(i%200000==0):
                print(i)

            line_split = line.strip().split(',')

            if len(line_split)==13:
                (ImageID, _, class_name, _, XMin, XMax, YMin, YMax, _, _, _, _, _) = line_split
            if len(line_split)==7:
                (ImageID,class_name,XMin,XMax,YMin,YMax,_) = line_split




            if i==1 or tmpimg['ImageID']!=ImageID:
                prob=0

                for bbox in tmpimg['bboxes']:
                    if probabilities[bbox['class']]==1:
                        prob=1
                        break
                    else:
                        prob=max(prob,probabilities[bbox['class']])

                tmpimg['maxProb']=prob

                if np.random.random()<prob:
                    for bbox in tmpimg['bboxes']:
                        classes_count[bbox['class']] += 1

                    all_data.append(tmpimg)

                tmpimg = {'filepath': None, 'ImageID': ImageID, 'width': None, 'height': None, 'bboxes': [],'imageset': 'train','maxProb':0}

            tmpimg['bboxes'].append({'class': class_name, 'XMin': float(XMin), 'XMax': float(XMax), 'YMin': float(YMin),'YMax': float(YMax)})

        all_data.append(tmpimg)

        return all_data, classes_count, class_mapping , eng_mapping ,probabilities ,classes_total

def get_data_allIDlist(input_path,descriptors_path,IDlist_path):

    def BinarySearch(lys, val):
        first = 0
        last = len(lys) - 1
        index = -1
        while (first <= last) and (index == -1):
            mid = (first + last) // 2
            if lys[mid] == val:
                index = mid
            else:
                if val < lys[mid]:
                    last = mid - 1
                else:
                    first = mid + 1
        return index

    IDlist=[]

    with open(IDlist_path, 'r') as f:
        for i, line in enumerate(f):
            IDlist.append(str(line.split('\n')[0]))

    IDlist.sort()

    all_data=[]
    class_mapping = {}
    classes_count = {}
    eng_mapping = {}
    probabilities = {}

    with open(descriptors_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split(',')
            (coded, engcoded) = line_split
            eng_mapping[coded] = engcoded
            class_mapping[coded] = i
            classes_count[coded] = 0
            probabilities[coded] = 1

    eng_mapping['bg'] = 'bg'
    class_mapping['bg'] = len(class_mapping)
    classes_count['bg'] = 0
    probabilities['bg'] = 0

    tmpimg = {'filepath':None , 'ImageID':None ,'width':None ,'height':None ,'bboxes':[] ,'imageset':'train','maxProb':0}

    with open(input_path,'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if(i%200000==0):
                print(len(IDlist))
                print(i)

            line_split = line.strip().split(',')

            if len(line_split)==13:
                (ImageID, _, class_name, _, XMin, XMax, YMin, YMax, _, _, _, _, _) = line_split
            if len(line_split)==7:
                (ImageID,class_name,XMin,XMax,YMin,YMax,_) = line_split

            if i==1 or tmpimg['ImageID']!=ImageID:

                if BinarySearch(IDlist, str(tmpimg['ImageID'])) != -1:
                    IDlist.remove(tmpimg['ImageID'])
                    tmpimg['maxProb']=1

                    for bbox in tmpimg['bboxes']:
                        classes_count[bbox['class']] += 1

                    all_data.append(tmpimg)

                tmpimg = {'filepath': None, 'ImageID': ImageID, 'width': None, 'height': None, 'bboxes': [],'imageset': 'train','maxProb':0}

            tmpimg['bboxes'].append({'class': class_name, 'XMin': float(XMin), 'XMax': float(XMax), 'YMin': float(YMin),'YMax': float(YMax)})

        if BinarySearch(IDlist, str(tmpimg['ImageID'])) != -1:
            IDlist.remove(tmpimg['ImageID'])
            tmpimg['maxProb'] = 1

            for bbox in tmpimg['bboxes']:
                classes_count[bbox['class']] += 1

            all_data.append(tmpimg)

        return all_data, classes_count, class_mapping , eng_mapping ,probabilities ,classes_count

def getparentdict(hierarchypath):
    with open(hierarchypath) as f:
        data = json.load(f)

    parentdict = {}
    parentdict['bg']=[]

    def process_subcategory(higher, dict):
        parentdict[dict['LabelName']] = higher

        if 'Subcategory' in dict.keys():
            for sub in dict['Subcategory']:
                tmp = copy.deepcopy(higher)
                tmp.append(dict['LabelName'])
                process_subcategory(copy.deepcopy(tmp), sub)

    for sub in data['Subcategory']:
        process_subcategory([], sub)

    return parentdict

