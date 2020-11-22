import os

def iouexist(dt_box, gt_box):
    if dt_box[0] > gt_box[2]:
        return False
    if gt_box[0] > dt_box[2]:
        return False
    if dt_box[3] < gt_box[1]:
        return False
    if dt_box[1] > gt_box[3]:
        return False
    return True


def iouCalculation(dt_box, gt_box):
    if iouexist:
        xA = max(float(dt_box[0]), float(gt_box[0]))
        yA = max(float(dt_box[1]), float(gt_box[1]))
        xB = min(float(dt_box[2]), float(gt_box[2]))
        yB = min(float(dt_box[3]), float(gt_box[3]))
        inter_region = (xB - xA + 1) * (yB - yA + 1)
        dt_box_region = (float(dt_box[2]) - float(dt_box[0]) + 1) * (float(dt_box[3]) - float(dt_box[1]) + 1)
        gt_box_region = (float(gt_box[2]) - float(gt_box[0]) + 1) * (float(gt_box[3]) - float(gt_box[1]) + 1)
        union_region = dt_box_region + gt_box_region - inter_region
        return inter_region / union_region
    else:
        return 0

def gt_boundingbox_info(file_path):
    with open(file_path) as file:
        # get bounding boxes information
        lines = file.readlines()
        # print(file_path)
        line = lines[0].strip()
        class_id, x1, y1, x2, y2 = line.split()

        return int(class_id), x1, y1, x2, y2

def boundingbox_info(file_path):
    with open(file_path) as file:
        # get bounding boxes information
        lines = file.readlines()
        class_id_list = []
        confidence_list = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for line in lines:
            line = line.strip()
            class_id, confidence, x1, y1, x2, y2 = line.split()
            class_id_list.append(class_id)
            confidence_list.append(confidence)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        index = confidence_list.index(max(confidence_list))


        confidence_final = float(confidence_list[index])
        det_class_id = class_id_list[index]
        x1_final = x1_list[index]
        y1_final = y1_list[index]
        x2_final = x2_list[index]
        y2_final = y2_list[index]

        return int(det_class_id), confidence_final, x1_final, y1_final, x2_final, y2_final

"""
Function: localization_eval
Description

********************************
  
"""
def localization_eval(det_folder_path, gt_folder_path):
    num_tp = 0
    num_fp = 0
    num_fn = 0
    num_tn = 0
    gt_name_list = os.listdir(gt_folder_path)
    for gt_name in gt_name_list:
        gt_file_path = os.path.join(gt_folder_path, gt_name)
        det_file_path = os.path.join(det_folder_path, gt_name)
        class_id, x1, y1, x2, y2 = gt_boundingbox_info(gt_file_path)
        det_class_id, det_conf_value, det_x1, det_y1, det_x2, det_y2 = boundingbox_info(det_file_path)
        iou = iouCalculation((det_x1, det_y1, det_x2, det_y2), (x1, y1, x2, y2))
        if det_conf_value > conf_threshold:
            # positive
            if iou > iou_threshold:
                num_tp = num_tp + 1
            else:
                num_fp = num_fp + 1
        else:
            if iou > iou_threshold:
                num_fn = num_fn + 1
            else:
                num_tn = num_tn + 1

    return num_tp, num_fp, num_fn, num_tn

"""
*********************************
                Ground Truth
            BBox        No_BBox
BBox        num_tp      num_fp
No_BBox     num_fn      num_tn
*********************************
"""

def result_show(num_tp, num_fp, num_fn, num_tn):

    if num_tp == 0:
        prec_show = 0
        rec_show = 0
    else:
        prec_show = num_tp / (num_tp + num_fp)
        rec_show = num_tp / (num_tp + num_fn)
    print("\n***********************************")
    print("                Ground Truth")
    print("            BBox        No_BBox")
    print("BBox        " + str(num_tp) + "         " + str(num_fp))
    print("No_BBox     " + str(num_fn) + "           " + str(num_tn))
    print("\nPrecision: ", prec_show)
    print("Recall: ", rec_show)
    print("***********************************")


conf_threshold = 0.35
iou_threshold = 0.5

currentPath = os.path.dirname(os.path.abspath(__file__))

det1_folder_name = 'detections1_tennis'
det1_folder_path = os.path.join(currentPath, det1_folder_name)
gt1_folder_name = 'groundtruths1_tennis'
gt1_folder_path = os.path.join(currentPath, gt1_folder_name)

num_tp_1, num_fp_1, num_fn_1, num_tn_1 = localization_eval(det1_folder_path, gt1_folder_path)
result_show(num_tp_1, num_fp_1, num_fn_1, num_tn_1)


det2_folder_name = 'detections2_tennis'
det2_folder_path = os.path.join(currentPath, det2_folder_name)
gt2_folder_name = 'groundtruths2_tennis'
gt2_folder_path = os.path.join(currentPath, gt2_folder_name)

num_tp_2, num_fp_2, num_fn_2, num_tn_2 = localization_eval(det2_folder_path, gt2_folder_path)
result_show(num_tp_2, num_fp_2, num_fn_2, num_tn_2)


det3_folder_name = 'detections3_tennis'
det3_folder_path = os.path.join(currentPath, det3_folder_name)
gt3_folder_name = 'groundtruths3_tennis'
gt3_folder_path = os.path.join(currentPath, gt3_folder_name)

num_tp_3, num_fp_3, num_fn_3, num_tn_3 = localization_eval(det3_folder_path, gt3_folder_path)
result_show(num_tp_3, num_fp_3, num_fn_3, num_tn_3)

num_tp_all = num_tp_1 + num_tp_2 + num_tp_3
num_fp_all = num_fp_1 + num_fp_2 + num_fp_3
num_fn_all = num_fn_1 + num_fn_2 + num_fn_3
num_tn_all = num_tn_1 + num_tn_2 + num_tn_3

result_show(num_tp_all, num_fp_all, num_fn_all, num_tn_all)
print("ALL:", num_tp_all + num_fp_all + num_fn_all + num_tn_all)