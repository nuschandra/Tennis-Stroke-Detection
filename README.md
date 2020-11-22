# Tennis-Stroke-Detection
This is the source code of the ITSS project. The title of the project is Automatic Detection of Tennis Strokes using Spatio-Temporal Localization.

Contributors:

Chandrashekar Viswanathan A0088591N
Cheng Yunfeng A0215320Y
Wang Ding (Jackson) A0216421U

In this project, we also initialize weights for the 2D CNN Stream using a YOLO model trained on the COCO dataset. We initialize weights for the 3D CNN Stream 
using pre-trained weights of ResNext model trained on the Kinetics dataset. These weights are used only for initialization and all layers of the model are fine-tuned as part of the end to end training process of the YOWO model.

These weights can be downloaded from: https://drive.google.com/drive/folders/1tE3oDN7EXqANtUfHiAuRL698Rl90mSmL
