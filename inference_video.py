
import argparse
import cv2
import torch.nn as nn
from torchvision import datasets, transforms

import InferenceDataset
from model import YOWO, get_fine_tuning_parameters
from utils import *



# Step 0 : Configuration
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', default=3, type=int, help='Number of classes (ucf101-24: 24, jhmdb-21: 21, tennis: 3)')
    parser.add_argument('--backbone_3d', default='resnext101', type=str, help='(resnext101 | resnet101 | resnet50 | resnet18 | mobilenet_2x | mobilenetv2_1x | shufflenet_2x | shufflenetv2_2x')
    parser.add_argument('--backbone_3d_weights', default='weights/resnext-101-kinetics.pth', type=str, help='Load pretrained weights for 3d_backbone')
    parser.add_argument('--freeze_backbone_3d', action='store_true', help='If true, 3d_backbone is frozen, else it is finetuned.')
    parser.set_defaults(freeze_backbone_3d=False)
    parser.add_argument('--backbone_2d', default='darknet', type=str, help='Currently there is only darknet19')
    parser.add_argument('--backbone_2d_weights', default='weights/yolo.weights', type=str, help='Load pretrained weights for 3d_backbone')
    parser.add_argument('--freeze_backbone_2d', action='store_true', help='If true, 2d_backbone is frozen, else it is finetuned.')
    parser.set_defaults(freeze_backbone_2d=False)
    parser.add_argument('--evaluate', action='store_true', help='If true, model is not trained, but only evaluated.')
    parser.set_defaults(evaluate=False)

    args = parser.parse_args()
    return args


# Test parameters
nms_thresh = 0.4
iou_thresh = 0.5

num_classes = 3
anchors = ['0.95878', '3.10197', '1.67204', '4.0040', '1.75482', '5.64937', '3.09299', '5.80857', '4.91803', '6.25225']
anchors = [float(i) for i in anchors]
num_anchors = 5

# Step 1: Set path for model and test dataset
model_path = r'F:\NUS\Courses\ISY5004\Practice Module\backup'
model_name = 'yowo_tennis_8f_best.pth'
Data_path = ''


# Step 2: load the model
opt = parse_opts()
model = YOWO(opt)

model       = model.cuda()
model       = nn.DataParallel(model, device_ids=None) # in multi-gpu case
model.seen  = 0
print(model)

print("===================================================================")
print('loading model {}'.format(model_path))
checkpoint = torch.load(os.path.join(model_path, model_name))
begin_epoch = checkpoint['epoch'] + 1
best_fscore = checkpoint['fscore']
model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# model.seen = checkpoint['epoch'] * nsamples
print("Loaded model fscore: ", checkpoint['fscore'])

# Step 3: Load test data
init_width = 224
init_height = 224
num_workers = 0
use_cuda = True
dataset_use = 'tennis'
batch_size = 2
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}


video_root = r'F:\Projects\YOWO-master\video'
video_name = 'ATPVideo_10fps'
output_video_name = video_name + '_10fps'
img_root_path = os.path.join(video_root, video_name + '_frames')
if not os.path.exists(img_root_path):
    os.makedirs(img_root_path)


"""
Function: extract frames from video
Input:
    video_root
    video_name
Frames_saved_folder_path:
    video_root/video_name + '_frames'
"""
def video2frames( video_root, video_name):
    video_path = os.path.join(video_root, video_name + '.mp4')
    vs = cv2.VideoCapture(video_path)
    name = 1
    output_folder = os.path.join(video_root, video_name + '_frames')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        output = frame.copy()
        file_name = str(name).zfill(4) + ".jpeg"
        name = name + 1
        cv2.imwrite(os.path.join(output_folder, file_name), output)
        if name % 10 == 0:
            print(name, " of frames extracted")
    vs.release()
    print("Have extracted frames from video!")

"""
Function: test
Input:
    A folder path which contains many frames
Output:
    position:
    
"""
def inference_frames_folder(img_root_path, detection_folder_name):
    test_loader = torch.utils.data.DataLoader(
        InferenceDataset.ListDataset(img_root_path, dataset_use=dataset_use, shape=(init_width, init_height),
                                     shuffle=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)

    conf_thresh_valid = 0.005
    total = 0.0
    proposals = 0.0
    correct = 0.0

    # Step 4: Get the result
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    for batch_idx, (frame, data) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                if dataset_use == 'tennis':
                    detection_path = os.path.join('tennis_detections', detection_folder_name, frame[i])
                    current_dir = os.path.join('tennis_detections', detection_folder_name)
                    if not os.path.exists('tennis_detections'):
                        os.mkdir('tennis_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = round(float(box[0] - box[2] / 2.0) * 1280.0)
                        y1 = round(float(box[1] - box[3] / 2.0) * 720.0)
                        x2 = round(float(box[0] + box[2] / 2.0) * 1280.0)
                        y2 = round(float(box[1] + box[3] / 2.0) * 720.0)

                        det_conf = float(box[4])
                        for j in range((len(box) - 5) // 2):
                            cls_conf = float(box[5 + 2 * j].item())

                            if type(box[6 + 2 * j]) == torch.Tensor:
                                cls_id = int(box[6 + 2 * j].item())
                            else:
                                cls_id = int(box[6 + 2 * j])
                            prob = det_conf * cls_conf

                            f_detect.write(
                                str(int(box[6]) + 1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                                    x2) + ' ' + str(y2) + '\n')

    print("Done")


detection_folder_name_list = ['TestVideo1_inference', 'TestVideo2_inference', 'TestVideo3_inference']
video_root = r'F:\Projects\YOWO-master\video'
video_name = 'TestVideo3_10fps'
img_root_path = os.path.join(video_root, video_name + '_frames')
detection_folder_name = detection_folder_name_list[2]


video2frames(video_root, video_name)
img_root_path = os.path.join(video_root, video_name + '_frames')
inference_frames_folder(img_root_path, detection_folder_name)
