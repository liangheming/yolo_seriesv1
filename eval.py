import torch
import os
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from datasets.coco import COCODataSets, coco_ids
from nets.yolov5 import YOLOv5
from commons.boxs_utils import clip_coords, xyxy2xywh
from utils.yolo_utils import non_max_suppression
from torch.utils.data.dataloader import DataLoader
from metrics.map import coco_map
from commons.augmentations import ScalePadding


@torch.no_grad()
def eval():
    device = torch.device("cuda:0")
    model = YOLOv5().to(device)
    weights = torch.load("weights/coco_yolov5_last.pth", map_location=device)['ema']
    model.load_state_dict(weights)
    model.eval()
    vdata = COCODataSets(img_root="/home/huffman/data/val2017",
                         annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                         img_size=640,
                         debug=False,
                         augments=False,
                         use_crowd=False,
                         remove_blank=False
                         )
    vloader = DataLoader(dataset=vdata,
                         batch_size=16,
                         num_workers=8,
                         collate_fn=vdata.collate_fn,
                         )
    predict_list = list()
    target_list = list()
    # self.model.eval()
    pbar = tqdm(vloader)
    for img_tensor, targets_tensor, _ in pbar:
        _, _, h, w = img_tensor.shape
        targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
        img_tensor = img_tensor.to(device)
        targets_tensor = targets_tensor.to(device)
        predicts = model(img_tensor)
        predicts = non_max_suppression(predicts,
                                       conf_thresh=0.001,
                                       iou_thresh=0.6,
                                       max_det=300,
                                       )
        for i, predict in enumerate(predicts):
            if predict is not None:
                clip_coords(predict, (h, w))
            predict_list.append(predict)
            targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 2:]
            target_list.append(targets_sample)
    mp, mr, map50, map = coco_map(predict_list, target_list)
    print("mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
          .format(mp * 100, mr * 100, map50 * 100, map * 100))


@torch.no_grad()
def gen_coco_predict(img_root="/home/huffman/data/val2017",
                     weight_path="weights/coco_yolov5_last.pth",
                     json_path="/home/huffman/data/annotations/instances_val2017.json"
                     ):
    weights = torch.load(weight_path)['ema']
    device = torch.device("cuda:0")
    model = YOLOv5().to(device)
    model.load_state_dict(weights)
    model.fuse().eval().half()
    basic_transform = ScalePadding(target_size=(640, 640),
                                   minimum_rectangle=True,
                                   padding_val=(103, 116, 123))
    coco = COCO(json_path)
    coco_predict_list = list()
    for img_id in tqdm(coco.imgs.keys()):
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        img, ratio, (left, top) = basic_transform.make_border(img)
        h, w = img.shape[:2]
        img_out = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).div(255.0).to(device).half()
        predicts = model(img_out)
        box = non_max_suppression(predicts,
                                  multi_label=True,
                                  iou_thresh=0.6,
                                  conf_thresh=0.001,
                                  merge=True)[0]
        if box is None:
            continue
        clip_coords(box, (h, w))
        # x1,y1,x2,y2,score,cls_id
        box[:, [0, 2]] = (box[:, [0, 2]] - left) / ratio[0]
        box[:, [1, 3]] = (box[:, [1, 3]] - top) / ratio[1]
        box = box.detach().cpu().numpy()
        # ret_img = draw_box(ori_img, box[:, [4, 5, 0, 1, 2, 3]], colors=coco_colors)
        # cv.imwrite(file_name, ret_img)
        pred_box = xyxy2xywh(box[:, :4])
        pred_box[:, :2] -= pred_box[:, 2:] / 2
        for p, b in zip(box.tolist(), pred_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
        # num += 1
        # if num > flag:
        #     break
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)


def coco_eavl():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO("/home/huffman/data/annotations/instances_val2017.json")  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes("predicts.json")  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    print(map, map50)


def check_coco_map():
    gen_coco_predict()
    coco_eavl()


if __name__ == '__main__':
    check_coco_map()
