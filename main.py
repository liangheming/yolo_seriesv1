# from processors.ddp_mix_processorv5 import COCODDPMixProcessor
from processors.ddp_mix_processorv4 import COCODDPMixProcessor
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 50003 main.py
if __name__ == '__main__':
    processor = COCODDPMixProcessor(cfg_path="config/coco_yolov4.yaml")
    # processor = COCODDPMixProcessor(cfg_path="config/coco_yolov5.yaml")
    processor.run()
