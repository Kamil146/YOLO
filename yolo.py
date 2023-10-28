import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import os

# Model
# colors for visualization

# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model = YOLO('yolov8m.pt')
# print(model.names)
# model.classes = [0]
# model.conf = 0.4
# Image
# name= r'C:\Users\Kamil\Downloads\Dataset_human_counting\DataSet_SEQUENCE_IPHONE_RGB_test2\Corridor1_RGB\00000000.png'
folder_dir = r'C:\Users\Kamil\Downloads\val2017\val2017'
classes = [0, 4, 8, 63, 77, 72, 16, 15, 14, 11]
values = [0 for i in range(len(classes))]
t = (classes, values)
count_dict = dict()

for i in range(len(classes)):
    count_dict[f'{classes[i]}']=values[i]
    i+=1
dfs=[]
for i, names in enumerate(sorted(os.listdir(folder_dir), key=lambda x: int(x[4:-4]))):

    name= r'{}\{}'.format(folder_dir,names)

    im = Image.open(name)

    model.predict(im, conf=0.4, classes=classes)
    # Inference

    results = model(im)
    # results.print()
    # results.show()

    for r in results:
        print(r.boxes.xyxy)
        print(r.boxes.conf)

        klasy=r.boxes.cls.cpu().tolist()
        prawdopodobienstwa=r.boxes.conf.cpu().tolist()
        bboxes=r.boxes.xyxy.cpu().tolist()

        # for e in classes:
        #     #print(f"klasa: {e} ilosc: {list.count(e)}")
        #     count_dict[f'{e}']=count_dict[f'{e}']+list.count(e)
        df1 = pd.DataFrame.from_dict({'nazwa zdjecia': names,
                                  'klasy': klasy,
                                  'prawdopodobienstwa': prawdopodobienstwa,
                                  "bbox'y": bboxes})
        dfs.append(df1)

    # if i>5:
    #     print("dfs", dfs)
    #     break;
ramki=pd.DataFrame()
for d in dfs:
    ramki=pd.concat([ramki,d])
ramki.to_csv("ramki.csv",sep=';')
        # im_array = r.plot()  # plot a BGR numpy array of predictions
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image

# plot_results(im, probas[keep], bboxes_scaled)
# print(count_dict)
# nazwa zdjecia klasa prawdop bboxy1 - bboxy4
# img1.jpg | [0,0,0] | [0.9, 0.8, 0.7] | [[9,9,19,19],[9,9,19,19],[9,9,19,19]]
#img1.jpg | [0] | [0.9,] | [[9,9,19,19],]
#img1.jpg | [0] | [0.8] | [[9,9,19,19],]

# klasa prawd bboxy1-bbox4
