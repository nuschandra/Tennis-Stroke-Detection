import os
import matplotlib.pyplot as plt


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
Confusion Metrix
***********************************************
                    Ground Truth
Prediction      Action      No_action
Action          num_tp      num_fp
No_action       num_fn      num_tn
***********************************************
"""
def result_show(num_tp, num_fp, num_fn, num_tn):
    # print('True positive:', num_tp)
    # print('False positive:', num_fp)
    # print('Flase negative:', num_fn)
    # print('True negative:', num_tn)
    # precision_action = num_tp / (num_tp + num_fp)
    # recall_action = num_tp / (num_tp + num_fn)
    # print("Action level precision:", precision_action)
    # print("Action level recall", recall_action)
    # accuracy_action = (num_tp + num_tn) / (num_tp + num_tn + num_fn + num_fp)
    # print("Action level accuracy:", accuracy_action)
    if num_tp == 0:
        prec_show = 0
        rec_show = 0
    else:
        prec_show = num_tp / (num_tp + num_fp)
        rec_show = num_tp / (num_tp + num_fn)
    print("\n***********************************************")
    print("                    Ground Truth")
    print("Prediction      Action      No_action")
    print("Action          " + str(num_tp) + "      " + str(num_fp))
    print("No_action       " + str(num_fn) + "      " + str(num_tn))
    print("\nPrecision: ", prec_show)
    print("Recall: ", rec_show)
    print("***********************************************")

conf_thre_list = [0.1, 0.2, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
rec_list = []
prec_list = []
for i in range(len(conf_thre_list)):
    conf_threshold = conf_thre_list[i]
    currentPath = os.path.dirname(os.path.abspath(__file__))

    det1_folder_name = 'detections1_tennis'
    det1_folder_path = os.path.join(currentPath, det1_folder_name)
    gt1_folder_name = 'groundtruths1_tennis'
    gt1_folder_path = os.path.join(currentPath, gt1_folder_name)
    num_tp_1, num_fp_1, num_fn_1, num_tn_1 = eval_action_level(det1_folder_path, gt1_folder_path, conf_threshold)
    # result_show(num_tp_1, num_fp_1, num_fn_1, num_tn_1)


    det2_folder_name = 'detections2_tennis'
    det2_folder_path = os.path.join(currentPath, det2_folder_name)
    gt2_folder_name = 'groundtruths2_tennis'
    gt2_folder_path = os.path.join(currentPath, gt2_folder_name)
    num_tp_2, num_fp_2, num_fn_2, num_tn_2 = eval_action_level(det2_folder_path, gt2_folder_path, conf_threshold)
    # result_show(num_tp_2, num_fp_2, num_fn_2, num_tn_2)


    det3_folder_name = 'detections3_tennis'
    det3_folder_path = os.path.join(currentPath, det3_folder_name)
    gt3_folder_name = 'groundtruths3_tennis'
    gt3_folder_path = os.path.join(currentPath, gt3_folder_name)
    num_tp_3, num_fp_3, num_fn_3, num_tn_3 = eval_action_level(det3_folder_path, gt3_folder_path, conf_threshold)
    # result_show(num_tp_3, num_fp_3, num_fn_3, num_tn_3)

    num_tp_all = num_tp_1 + num_tp_2 + num_tp_3
    num_fp_all = num_fp_1 + num_fp_2 + num_fp_3
    num_fn_all = num_fn_1 + num_fn_2 + num_fn_3
    num_tn_all = num_tn_1 + num_tn_2 + num_tn_3

    if num_tp_all == 0:
        prec = 0
        rec = 0
    else:
        prec = num_tp_all / (num_tp_all + num_fp_all)
        rec = num_tp_all / (num_tp_all + num_fn_all)

    prec_list.append(prec)
    rec_list.append(rec)

    result_show(num_tp_all, num_fp_all, num_fn_all, num_tn_all)


print("Precision list:", prec_list)
print("Recall list:", rec_list)
plt.plot(rec_list, prec_list, label='Presion')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision x Recall curve')
plt.grid()
plt.savefig(os.path.join(currentPath, 'Precision x Recall curve.png'))
plt.show()