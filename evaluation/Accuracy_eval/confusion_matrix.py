import os


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
Function: eval_action_level
Description: We want to evaluate if the model can find the action.
Input: detection folder path, gt_folder_path, conf_threshold
Output: 4 values
************************************************
            | Ground Truth    |  
            Action    No_Action  
Action      num_TP    num_FP  
No_Action   num_FN    num_TN   
************************************************
Step 1: get the detection file name
Step 2: Get the bounding box with maximun confidence value
Step 3: If conf_value > conf_threshold, which means the detection is positive.
    Step 3_1_1: If so, becasue the detection is positive, we need to find out if there is a ground truth file.
    Step 3-1_2: If there exists, for action_level, it is a true positive prediction.
    Step 3_1-3: If there not exists, it is a false positive prediction

    Step 3_2_1: If conf_value < conf_threshold. It is a negative prediction. Need to check if there exists ground truth.
    Stpe 3_2_2: If ground truth exists, it is a false negative prediction
Step4: Return all values.
"""
def eval_action_level(det_folder_path, gt_folder_path, conf_threshold):
    num_tp = 0
    num_fp = 0
    num_fn = 0
    num_tn = 0
    det_name_list = os.listdir(det_folder_path)
    for det_name in det_name_list:
        det_file_path = os.path.join(det_folder_path, det_name)
        gt_file_path = os.path.join(gt_folder_path, det_name)
        class_id, conf_value, x1, y1, x2, y2 = boundingbox_info(det_file_path)
        if conf_value > conf_threshold:
            if os.path.exists(gt_file_path):
                num_tp = num_tp + 1
            else:
                num_fp = num_fp + 1
        else:
            if os.path.exists(gt_file_path):
                num_fn = num_fn + 1
            else:
                num_tn = num_tn + 1
    return num_tp, num_fp, num_fn, num_tn


"""
Function: eval_class_level()
Description: Because we train the model only on the frames with action. So we want to evaluate it in the same way.
Input: detection folder path, gt_folder_path, conf_threshold
Output: 16 values
************************************************
            |       Ground Truth          |  
            Forehand    Backhand    Serve  
Forehand    num_FF      num_BF      num_SF  
Backhand    num_FB      num_BB      num_SB
Serve       num_FS      num_BS      num_SS
No_action   num_FN      num_BN      num_SN
************************************************
For example, num_FB means that in ground truth label, it is a Forehand but in detection folder, it is a Backhand.
This time we should start from the ground truth folder
 
class_id
1: serve
2: forehand
3: backhand
 
Step 1: Get the name in ground truth folder
Step 2: Get the ground truth information and bounding box information
Step 3: Based on the class id we can use 3 if for three class

    Step 3_1: For serve, which means class_id == 1. In this case we will change num_SF, num_SB, num_SS, num_SN
    Step 3-2: First check if there is the confidencen value of bounding box is larger than threshold, if nor, no action.
    

"""
def eval_class_level(det_folder_path, gt_folder_path, conf_threshold):
    # initialization_eval_class_level()
    num_FF = 0
    num_FB = 0
    num_FS = 0
    num_FN = 0
    num_BF = 0
    num_BB = 0
    num_BS = 0
    num_BN = 0
    num_SF = 0
    num_SB = 0
    num_SS = 0
    num_SN = 0
    gt_name_list = os.listdir(gt_folder_path)
    for gt_name in gt_name_list:
        gt_file_path = os.path.join(gt_folder_path, gt_name)
        det_file_path = os.path.join(det_folder_path, gt_name)
        class_id, x1, y1, x2, y2 = gt_boundingbox_info(gt_file_path)
        det_class_id, det_conf_value, det_x1, det_y1, det_x2, det_y2 = boundingbox_info(det_file_path)
        # For serve
        if class_id == 1:
            if det_conf_value <= conf_threshold:
                num_SN = num_SN +1
            else:
                if det_class_id == 1:
                    num_SS = num_SS + 1
                elif det_class_id == 2:
                    num_SF = num_SF + 1
                elif det_class_id == 3:
                    num_SB = num_SB + 1
        elif class_id == 2:
            if det_conf_value <= conf_threshold:
                num_FN = num_FN + 1
            else:
                if det_class_id == 1:
                    num_FS = num_FS + 1
                elif det_class_id == 2:
                    num_FF = num_FF + 1
                elif det_class_id == 3:
                    num_FB = num_FB + 1
        elif class_id == 3:
            if det_conf_value <= conf_threshold:
                num_BN = num_BN + 1
            else:
                if det_class_id == 1:
                    num_BS = num_BS + 1
                elif det_class_id == 2:
                    num_BF = num_BF + 1
                elif det_class_id == 3:
                    num_BB = num_BB + 1
    return num_BB, num_BF, num_BS, num_BN, num_FB, num_FF, num_FS, num_FN, num_SB, num_SF, num_SS, num_SN



def result_show(num_tp, num_fp, num_fn, num_tn):
    print('True positive:', num_tp)
    print('False positive:', num_fp)
    print('Flase negative:', num_fn)
    print('True negative:', num_tn)

    precision_action = num_tp / (num_tp + num_fp)
    recall_action = num_tp / (num_tp + num_fn)
    print("Action level precision:", precision_action)
    print("Action level recall", recall_action)

    accuracy_action = (num_tp + num_tn) / (num_tp + num_tn + num_fn + num_fp)
    print("Action level accuracy:", accuracy_action)


def show_eval_class_level(num_BB, num_BF, num_BS, num_BN, num_FB, num_FF, num_FS, num_FN, num_SB, num_SF, num_SS, num_SN):
    """
    ************************************************
                |       Ground Truth          |
                Forehand    Backhand    Serve
    Forehand    num_FF      num_BF      num_SF
    Backhand    num_FB      num_BB      num_SB
    Serve       num_FS      num_BS      num_SS
    No_action   num_FN      num_BN      num_SN
    ************************************************
    """
    print("************************************************")
    print("            |       Ground Truth          |     ")
    print("            Forehand    Backhand    Serve  ")
    print("Forehand    " + str(num_FF) + "           " + str(num_BF) + "            " + str(num_SF))
    print("Backhand    " + str(num_FB) + "           " + str(num_BB) + "            " + str(num_SB))
    print("Serve       " + str(num_FS) + "           " + str(num_BS) + "            " + str(num_SS))
    print("Noaction    " + str(num_FN) + "           " + str(num_BN) + "            " + str(num_SN))
    print("************************************************")



conf_threshold = 0.35
currentPath = os.path.dirname(os.path.abspath(__file__))

det_folder_name = 'detections_tennis'
det_folder_path = os.path.join(currentPath, det_folder_name)
if not os.path.exists(det_folder_path):
    print("No detection folder")

gt_folder_name = 'groundtruths_tennis'
gt_folder_path = os.path.join(currentPath, gt_folder_name)
if not os.path.exists(gt_folder_path):
    print("No ground truth folder")

# num_tp, num_fp, num_fn, num_tn = eval_action_level(det_folder_path, gt_folder_path, conf_threshold)
# result_show(num_tp, num_fp, num_fn, num_tn)

# num_BB, num_BF, num_BS, num_BN, num_FB, num_FF, num_FS, num_FN, num_SB, num_SF, num_SS, num_SN = eval_class_level(det_folder_path, gt_folder_path, conf_threshold)
# show_eval_class_level(num_BB, num_BF, num_BS, num_BN, num_FB, num_FF, num_FS, num_FN, num_SB, num_SF, num_SS, num_SN)

det1_folder_name = 'detections1_tennis'
det1_folder_path = os.path.join(currentPath, det1_folder_name)
gt1_folder_name = 'groundtruths1_tennis'
gt1_folder_path = os.path.join(currentPath, gt1_folder_name)
num_BB_1, num_BF_1, num_BS_1, num_BN_1, num_FB_1, num_FF_1, num_FS_1, num_FN_1, num_SB_1, num_SF_1, num_SS_1, num_SN_1 = eval_class_level(det1_folder_path, gt1_folder_path, conf_threshold)
show_eval_class_level(num_BB_1, num_BF_1, num_BS_1, num_BN_1, num_FB_1, num_FF_1, num_FS_1, num_FN_1, num_SB_1, num_SF_1, num_SS_1, num_SN_1)

det2_folder_name = 'detections2_tennis'
det2_folder_path = os.path.join(currentPath, det2_folder_name)
gt2_folder_name = 'groundtruths2_tennis'
gt2_folder_path = os.path.join(currentPath, gt2_folder_name)
num_BB_2, num_BF_2, num_BS_2, num_BN_2, num_FB_2, num_FF_2, num_FS_2, num_FN_2, num_SB_2, num_SF_2, num_SS_2, num_SN_2 = eval_class_level(det2_folder_path, gt2_folder_path, conf_threshold)
show_eval_class_level(num_BB_2, num_BF_2, num_BS_2, num_BN_2, num_FB_2, num_FF_2, num_FS_2, num_FN_2, num_SB_2, num_SF_2, num_SS_2, num_SN_2)


det3_folder_name = 'detections3_tennis'
det3_folder_path = os.path.join(currentPath, det3_folder_name)
gt3_folder_name = 'groundtruths3_tennis'
gt3_folder_path = os.path.join(currentPath, gt3_folder_name)
num_BB_3, num_BF_3, num_BS_3, num_BN_3, num_FB_3, num_FF_3, num_FS_3, num_FN_3, num_SB_3, num_SF_3, num_SS_3, num_SN_3 = eval_class_level(det3_folder_path, gt3_folder_path, conf_threshold)
show_eval_class_level(num_BB_3, num_BF_3, num_BS_3, num_BN_3, num_FB_3, num_FF_3, num_FS_3, num_FN_3, num_SB_3, num_SF_3, num_SS_3, num_SN_3)

num_BB_all = num_BB_1 + num_BB_2 + num_BB_3
num_BS_all = num_BS_1 + num_BS_2 + num_BS_3
num_BF_all = num_BF_1 + num_BF_2 + num_BF_3
num_BN_all = num_BN_1 + num_BN_2 + num_BN_3


num_FB_all = num_FB_1 + num_FB_2 + num_FB_3
num_FS_all = num_FS_1 + num_FS_2 + num_FS_3
num_FF_all = num_FF_1 + num_FF_2 + num_FF_3
num_FN_all = num_FN_1 + num_FN_2 + num_FN_3


num_SB_all = num_SB_1 + num_SB_2 + num_SB_3
num_SS_all = num_SS_1 + num_SS_2 + num_SS_3
num_SF_all = num_SF_1 + num_SF_2 + num_SF_3
num_SN_all = num_SN_1 + num_SN_2 + num_SN_3

show_eval_class_level(num_BB_all, num_BF_all, num_BS_all, num_BN_all, num_FB_all, num_FF_all, num_FS_all, num_FN_all, num_SB_all, num_SF_all, num_SS_all, num_SN_all)
print("ALL:", num_BB_all + num_BF_all + num_BS_all + num_BN_all + num_FB_all + num_FF_all + num_FS_all + num_FN_all + num_SB_all + num_SF_all + num_SS_all+ num_SN_all)
