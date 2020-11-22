import os
import cv2


"""
Description: Draw the bounding box with highest value on the image

Input:
    img_path: The absolute path of image
    label_path: The absolute path of a .txt file which contains information about bounding boxes.
                The information should be displayed in such a way: (class_id, confidence, x1, y1, x2, y2)
                class_id: action id
                confidence: To which degree you believe this is true
                x1, y1: the top-left coordinate of bounding box
                x2, y2: the bottom-right coordinate of bounding box
Output:
    image with highest confidence value bounding box
    
Attention: 
    The confidence value is required for this function, so it's not suitable for ground truth label.
"""
def visualize(img_path, label_path):

    frame = cv2.imread(img_path)
    img = frame.copy()

    with open(label_path) as file:
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
        if len(confidence_list) == 0:
            return img
        else:
            index = confidence_list.index(max(confidence_list))
            if (float(confidence_list[index]) < 0.45):
                return img
            x1 = int(x1_list[index])
            y1 = int(y1_list[index])
            x2 = int(x2_list[index])
            y2 = int(y2_list[index])
            label = int(class_id_list[index])
            print(class_id_name[label-1])
            txtlbl = "{}:{:.2f}".format(class_id_name[label-1], float(confidence_list[index]))

            txtsize = cv2.getTextSize(txtlbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            bsize = txtsize[0]  # extract the width and height
            bsline = txtsize[1] # extract the height of baseline

            # Draw the bounding box, put up the text
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(img, (x1 - 1, y1), (x1 + bsize[0], y1 + bsize[1] + bsline), (0, 255, 0), -1)
            cv2.putText(img, txtlbl, (x1 - 1, y1 + bsize[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            return img



def visualizeGT(img_path, label_path):
    frame = cv2.imread(img_path)
    img = frame.copy()

    with open(label_path) as file:
        lines = file.readlines()
        class_id_list = []
        confidence_list = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for line in lines:
            line = line.strip()
            class_id, x1, y1, x2, y2 = line.split()
            class_id_list.append(class_id)
            # confidence_list.append(confidence)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        # index = confidence_list.index(max(confidence_list))
        index = 0
        x1 = int(float(x1_list[index]))
        y1 = int(float(y1_list[index]))
        x2 = int(float(x2_list[index]))
        y2 = int(float(y2_list[index]))

        txtlbl = "{}:{:.2f}".format(class_id[0], float(1))

        txtsize = cv2.getTextSize(txtlbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        bsize = txtsize[0]  # extract the width and height
        bsline = txtsize[1] # extract the height of baseline

        # Draw the bounding box, put up the text
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.rectangle(img, (x1-1, y1), (x1 + bsize[0], y1 + bsize[1] + bsline), (0, 255, 0), -1)
        cv2.putText(img, txtlbl, (x1 - 1, y1 + bsize[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img


# TODO::Version_5: For Test tennis
class_id_name = ['Serve', 'Forehand', 'Backhand']

label_root_path = 'F:/Projects/YOWO-master/tennis_detections/TestVideo3_inference/'
# img_root_path = 'F:/Data/video/tennis/rgb-images/Tennis2Frames'
img_root_path = 'F://Projects/YOWO-master/video/TestVideo3_10fps_frames'
video_output_path = 'F:/Projects/YOWO-master/show'

# Step 3: Set parameters for video
fps = 10
W = 1280
H = 720
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter(video_output_path, fourcc, fps, (W, H), True)

video_output_path = os.path.join(video_output_path, 'TestVideo3_2_10fps_detection.avi')
writer = cv2.VideoWriter(video_output_path, fourcc, fps, (W, H), True)
target_path = os.path.join(label_root_path)
name_list = os.listdir(target_path)
print(len(name_list))
for label_name in name_list[:2000]:
    img_path = os.path.join(img_root_path, label_name[0:4] + '.jpeg')
    print(img_path)
    label_path = os.path.join(target_path, label_name)
    img = visualize(img_path, label_path)
    writer.write(img)
print("Closing")
writer.release()

# name_list = os.listdir(aim_path)
# print(len(name_list))