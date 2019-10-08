from test_frcnn import testimages
import copy
import os

parms=[0.04,0.36,0.82]
parmstoupdate=[0]
update_vec=[0.0,0.0,0.0]
parmshistory=[]
epoch=0
curr_mAP=0.0

last_mAP = testimages(bbox_threshold=parms[0],
                          non_max_sup_thresh=parms[1],
                          rpn_to_roi_overlap_thresh=parms[2])

print(last_mAP)
b = os.path.getsize("/home/adam/Desktop/OIDproject/results_imgs/predictions.csv")
mb = b/1000000
print(mb)

def normalize(num):
    num=max(num,0.0001)
    num=min(num,0.9999)
    return num

while(True):
    for idx in parmstoupdate:
        tmpparms=copy.deepcopy(parms)
        tmpparms[idx] = normalize(tmpparms[idx] + 0.01)
        curr_mAP = testimages(bbox_threshold = tmpparms[0],
                              non_max_sup_thresh = tmpparms[1],
                              rpn_to_roi_overlap_thresh = tmpparms[2])

        if curr_mAP>last_mAP:
            update_vec[idx]= 0.01
        else:
            update_vec[idx]=-0.01

    for idx in parmstoupdate:
        parms[idx]=normalize(parms[idx] + update_vec[idx])

    last_mAP = testimages(bbox_threshold=parms[0],
                          non_max_sup_thresh=parms[1],
                          rpn_to_roi_overlap_thresh=parms[2])

    b = os.path.getsize("/home/adam/Desktop/OIDproject/results_imgs/predictions.csv")
    mb = b/1000000

    parmshistory.append(["{:.3f}".format(parms[0]),"{:.3f}".format(parms[1]),"{:.3f}".format(parms[2]),"{:.3f}".format(last_mAP),"{:.3f}".format(mb),epoch])

    print("#############################")
    print("parmshistory:")
    print(parmshistory)
    print("parms:{}".format(parms))
    print("epoch:{}".format(epoch))
    print("mAP:{}".format(last_mAP))

    epoch+=1



