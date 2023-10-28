from COCOParser import COCOParser
import pandas as pd
from AP import mean_average_precision
coco_annotations_file = "instances_val2017 — kopia.json"
coco_images_dir = r"C:\Users\Kamil\Downloads\val2017\val2017"
coco = COCOParser(coco_annotations_file, coco_images_dir)
classes = [1, 5, 9, 88, 82, 73, 18, 17, 16, 13]
imgs_id = coco.get_imgIds()
annotations_id = coco.get_annIds(imgs_id)
annotations = coco.load_anns(annotations_id)
images = coco.coco['images']
print(annotations[0]['category_id'])
print(images[0]['file_name'])
# nazwa klasa bboxy - gt
# nazwa klasa prawd boxy - detections

dflist = []
images_names = []
for i in images:
    images_names.append(i['file_name'])
i = 0

for row in range(len(annotations)):

    names = annotations[row]['image_id']
    for i in range(12 - len(str(names))):
        names = '0' + str(names)
    names = str(names) + '.jpg'
    klasy = annotations[row]['category_id']
    bboxes = [annotations[row]['bbox']]
    if klasy in classes:
        if klasy == 1:
            k = 0
        if klasy == 5:
            k = 4
        if klasy == 9:
            k = 8
        if klasy == 88:
            k = 77
        if klasy == 82:
            k = 72
        if klasy == 73:
            k = 63
        if klasy == 18:
            k = 16
        if klasy == 17:
            k = 15
        if klasy == 16:
            k = 14
        if klasy == 13:
            k = 11
        df1 = pd.DataFrame.from_dict({'nazwa zdjecia': names,
                                      'klasy': k,
                                      "bbox'y": bboxes})
        dflist.append(df1)
    image_id = annotations[row]['image_id']

ramki = pd.DataFrame()
for d in dflist:
    ramki = pd.concat([ramki, d])
ramki.sort_values(by=['nazwa zdjecia'], inplace=True)
ramki.to_csv("ramki_gt.csv", sep=';')

ramki_detection = pd.read_csv('ramki.csv', sep=';')
print(ramki_detection)
print(ramki)

#print(mean_average_precision(pred_boxes,GT)) - somewhere in the future
# gt [ [nazwa/idx klasa bboxy]
# [nazwa/idx klasa bboxy]
# ]
