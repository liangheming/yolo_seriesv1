# YOLO_series
该仓库是[YOLOv5](https://github.com/ultralytics/yolov5) 的一个重写版本，我们参考ultralytics版本的YOLO实现，按照自己的编码习惯对整个代码
库进行了重新整理，目的是为了更清晰的理解[YOLOv5](https://github.com/ultralytics/yolov5)的原理。同时我们对重写的代码进行训练，各种超参数的设置与
原仓库一致，模型的性能比原仓库略微差一点(YOLOv5s mAP36.7而原实现 mAP37.0，我们怀疑是训练时我们去掉了coco数据集中crowed数据，以及空白的数据)，
另外我们也参考了[YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) 将其中的一部分yolov4的模型集成到了该仓库中。该仓库中的不支持使用yaml进行
模型设计，需要完全从代码开始重新构建，因此在工程上缺少了一些灵活性，但是适合与初学者理解从代码角度理解模型的结构。
## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
## result
我们使用原仓库中的超参数设置，使用2块GPU,YOLOv5s版本的batch_size=64(每个GPU，batch_size=32),最终的模型性能为mAP 36.7


## training
目前，我们只支持COCO数据集的训练，感兴趣的同学可以自行参考datasets/coco.py下的代码，进行自定义数据集的训练(我相信我们的代码逻辑是比较清晰的，重写应该没有什么难度)

### COCO
* modify main.py (modify config file path)
```python
from processors.ddp_mix_processorv5 import COCODDPMixProcessor
if __name__ == '__main__':
    processor = COCODDPMixProcessor(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: coco_yolov5
data:
  train_annotation_path: .../annotations/instances_train2017.json
  val_annotation_path: .../annotations/instances_val2017.json
  train_img_root: .../train2017
  val_img_root:.../val2017
  img_size: 640
  batch_size: 20
  num_workers: 8
  debug: False
  remove_blank: False
  use_crowd: False

model:
  num_cls: 80
  anchors: [[10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
  strides: [8, 16, 32]
  scale_name: m
  freeze_bn: False

hyper_params:
  iou_type: ciou
  multi_scale: [640]

optim:
  optimizer: SGD
  lr: 0.01
  momentum: 0.937
  alpha: 0.2
  gamma: 1.0
  warm_up_epoch: 3
  weight_decay: 0.0005
  epochs: 300
  sync_bn: True
val:
  interval: 1
  weight_path: weights
  conf_thresh: 0.001
  iou_thresh: 0.6
  max_det: 300
gpus: 4,5
```
* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] EMA(Exponential Moving Average)
- [x] Mixed Precision Training (supported by apex)
- [x] Sync Batch Normalize
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support

## Reference
1. ultralytics/yolov5 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
2. WongKinYiu/PyTorch_YOLOv4 [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
