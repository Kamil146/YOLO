import math
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl+1]}: {p[cl]:0.4f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
#model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-e6e.pt',
#                        force_reload=True, trust_repo=True)
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval();

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# im = Image.open(requests.get(url, stream=True).raw)

#pathImg = r'C:\Users\Kamil\Downloads\YOLOv3 person detection\DataSet_SEQUENCE_IPHONE_RGB_test2\Corridor1_RGB\00000000.png'

tab = ['Corridor1_RGB', 'Corridor2_RGB', 'Corridor3_RGB', 'D3A_RGB', 'D7_RGB', 'F102_RGB', 'F104_RGB', 'F107_RGB',
       'F105_RGB']


for j in tab:

    img_dir = r'C:\Users\Kamil\Downloads\YOLOv3 person detection\DataSet_SEQUENCE_PiRobot_RGB_test2'
    folder_dir = os.path.join(img_dir, j)

    for i, names in enumerate(sorted(os.listdir(folder_dir), key=lambda x: int(x[4:-4]))):


        name= r'{}\{}\{}'.format(img_dir, j, names)
        im = Image.open(name)
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0).to(device)

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, 1:2].to(device)
        keep = probas.max(-1).values > 0.4

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size).to(device)
        #plot_results(im, probas[keep], bboxes_scaled)
        save_path =r'C:\Users\Kamil\PycharmProjects\cuda\test'
        # save_path = os.path.dirname(path)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        path = r'{}\{}_{}.txt'.format(save_path,j, names[:-4])
        with open(path, 'w') as f:
            for y, x in zip(probas[keep].tolist(),bboxes_scaled.tolist()):
                y[0] = round(y[0],7)
                x = [round(val) for val in x]
                line = 'person {} {} {} {} {}\n'.format(y[0], x[0], x[1], x[2], x[3])
                f.write(line)


