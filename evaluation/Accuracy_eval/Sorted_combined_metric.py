import os
import matplotlib.pyplot as plt
import numpy as np

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

# Step1: Using dictionary{'name': 'score'} save information at frame level
# Step2: Using a list to sort

conf_threshold = 0.35
iou_threshold = 0.5


current_folder = 'F:/Projects/YOWO-master/evaluation/Accuracy_eval'
det_folder_name = 'detections3_tennis'
gt_folder_name = 'groundtruths3_tennis'

det_folder_path = os.path.join(current_folder, det_folder_name)
gt_folder_path = os.path.join(current_folder, gt_folder_name)

num_frames = len(os.listdir(det_folder_path))
num_gt = len(os.listdir(gt_folder_path))

det_dict = {}
det_dict_list = []

for det_frame_name in os.listdir(det_folder_path):
    det_frame_path = os.path.join(det_folder_path, det_frame_name)
    with open(det_frame_path) as detfile:
        # det_dict['name'] = det_frame_name
        det_lines = detfile.readlines()
        det_line = det_lines[0]
        det_line = det_line.rstrip()
        _, conf, _, _, _, _ = det_line.split()
        det_dict['name'] = det_frame_name
        det_dict['conf_value'] = conf
        det_dict_list.append(det_dict.copy())
# print(det_dict_list)
sorted_det_dict_list = sorted(det_dict_list, key=lambda k: k['conf_value'], reverse=True)
# print(sorted_det_dict_list)

class_accuracy_num = 0
localize_accuracy_num = 0

tp_list = np.zeros(num_frames)
fp_list = np.zeros(num_frames)

for i in range(num_frames):
    class_match = False
    localize_match = False
    name_of_frame = sorted_det_dict_list[i]['name']
    with open(os.path.join(det_folder_path, name_of_frame)) as detfile:
        # get bounding boxes information
        det_lines = detfile.readlines()
        class_id_list = []
        confidence_list = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for line in det_lines:
            line = line.strip()
            class_id, confidence, x1, y1, x2, y2 = line.split()
            class_id_list.append(class_id)
            confidence_list.append(confidence)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        index = confidence_list.index(max(confidence_list))
        if ( float( confidence_list[index] ) > 0.25):
            # If the maximum exist and > threshold then should be positive.
            print('Detection Positive')
            x1 = x1_list[index]
            y1 = y1_list[index]
            x2 = x2_list[index]
            y2 = y2_list[index]
            det_class_id = class_id_list[index]
            # print(class_id_name[label - 1])

            gt_name = os.path.join(gt_folder_path, name_of_frame)
            if os.path.exists(gt_name):
                print("GT_file exists")
                # calculate IOU
                # So far, we have gotten information from detection file, we still need that from ground truth file
                # Get bounding box from ground truth file
                with open(gt_name) as gtfile:
                    gt_lines = gtfile.readlines()
                    line = gt_lines[0].strip()
                    gt_class_id, gt_x1, gt_y1, gt_x2, gt_y2 = line.split()
                    # Have gotten ground truth bouning box information.
                    # First calculate the classification accuracy
                    if det_class_id == gt_class_id:
                        # TODO:: TP for classification
                        class_match = True
                        class_accuracy_num = class_accuracy_num + 1
                    else:
                        # class id not same. False positive.
                        fp_list[i] = 1
                    # Second calculate the localization accuracy
                    dt_box = [x1, y1, x2, y2]
                    gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
                    # 1) Calculate IOU
                    #   a) If there is IoU
                    #   b) True, calculate Iou
                    #   c) False, returen 0
                    iou = iouCalculation( dt_box, gt_box)
                    # 2) Justify if IOU > threshold
                    if iou > iou_threshold:
                        # TODO:: TP for localization
                        localize_match = True
                        localize_accuracy_num = localize_accuracy_num + 1
                    else:
                        fp_list[i] = 1

                    if class_match and localize_match:
                        tp_list[i] = 1

            else:
                fp_list[i] = 1

        else:
            print(" Negative detection")
            # Check if the bounding box is negative also.
            if os.path.exists(gt_name):
                print('Ground truth bounding box exist, so this one should be False Negative')
            else:
                # TODO:: detection for classification and localization is right
                print(' No GT bbox, True negative')
                class_accuracy_num = class_accuracy_num + 1
                localize_accuracy_num = localize_accuracy_num + 1


class_accuracy = class_accuracy_num / num_frames
localize_accuracy = localize_accuracy_num / num_frames
print(class_accuracy)
acc_FP = np.cumsum(fp_list)
acc_TP = np.cumsum(tp_list)

rec = acc_TP / num_gt
print('rec:',rec)
prec = np.divide(acc_TP, (acc_FP + acc_TP))

# I guess append 0 and 1 for ploting
mrec = []
mrec.append(0)
[mrec.append(e) for e in rec]
mrec.append(1)
mpre = []
mpre.append(0)
[mpre.append(e) for e in prec]
mpre.append(0)
for i in range(len(mpre) - 1, 0, -1):
    mpre[i - 1] = max(mpre[i - 1], mpre[i])

ii = []
for i in range(len(mrec) - 1):
    if mrec[1:][i] != mrec[0:-1][i]:
        ii.append(i + 1)
ap = 0
for i in ii:
    ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

# return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
plt.plot(rec, prec, label='Presion')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision x Recall curve')
plt.legend(shadow=True)
plt.grid()


# fig_path = os.path.join(current_folder, 'TestVideo1_results')
# plt.savefig(os.path.join(fig_path, 'Curve.png'))
# plt.show()
# plt.pause(0.05)





