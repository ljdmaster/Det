import numpy as np



def count(pred_boxs, real_boxs):
    count = 0
    index = []
    for i in range(pred_boxs.shape[0]):
        ious=[]
        for j in range(real_boxs.shape[0]):
            if j in index:
                ious.append(0.)
            else: 
                ious.append(iou(pred_boxs[i], real_boxs[j]))
        max_iou  = max(ious)
        if max_iou>0.4:
            index.append(ious.index(max(ious)))
            count +=1
    return count



def iou(b1, b2):
    # b1
    b1_xy = b1[:2]
    b1_wh = b1[2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    
    # b2
    b2_xy = b2[:2]
    b2_wh = b2[2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = max(list(b1_mins), list(b2_mins))
    intersect_maxes = min(list(b1_maxes), list(b2_maxes))
    intersect_wh = max(list(np.array(intersect_maxes)-np.array(intersect_mins)),[0., 0.])
    intersect_area = intersect_wh[0] * intersect_wh[1]
    iou = intersect_area/(b2_wh[0]*b2_wh[1])

    return iou



if __name__=="__main__":
    b1 = np.array([[10,10,100,100]])
    b2 = np.array([[100, 100, 100, 100]])
    count  = count(b1, b2)
    print(count)

