import torch
import numpy as np
import pandas as pd
import random
import argparse, configparser
from collections import Counter
from iou import intersection_over_union
import matplotlib.pyplot as plt


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, score_threshold=0,classes=[0.0, 4.0, 8.0, 63.0, 77.0, 72.0, 16.0, 15.0, 14.0, 11.0]):
    average_precisions = []
    average_f1 = []
    ground_truths = []
    F1=0
    # used for numerical stability later on
    epsilon = 1e-6
    for c in classes:
        average_precision = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        if score_threshold>0:
            detections = [tablica for tablica in pred_boxes if tablica[1] > score_threshold]
        else:
            detections = pred_boxes

        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        # sort by box probabilities which is index 1
        detections.sort(key=lambda x: x[1], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):

                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[2:]),

                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precision = precisions[-1:]
        recall = recalls[-1:]
        F1 = 2/(1/precision + 1/recall)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # torch.trapz for numerical integration
        average_precision=torch.trapz(precisions, recalls)
        # plt.plot(recalls, precisions, label='Precision-Recall Curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.legend()
        # plt.show()
        average_precisions.append(torch.trapz(precisions, recalls))
        average_f1.append(F1)
    return sum(average_precisions) / len(average_precisions), sum(average_f1)/len(F1)