# Detectron2



- not a model but a framework

- https://github.com/facebookresearch/detectron2/

- https://detectron2.readthedocs.io/

  

## Key components

- demo

  - demo.py

- detectron2

  - config

    - config
      - CfgNode

  - data

    - catalog
      - _MetadataCatalog
      - _DatasetCatalog
    - build_detection_test_loader()

  - engine

    - defaults

      - DefaultPredictor

    - DefaultTrainer

      - resume_or_load()

      - train()

  - evaluation

    - COCOEvaluator
    - inference_on_dataset()

  - utils

    - visualizer
      - Visualizer
        - draw_instance_predictions()

    

  ## Model APIs

  - https://detectron2.readthedocs.io/en/latest/tutorials/models.html
  - input: dict
    - "image"
    - "height"
    - "width"
    - "instances"
      - "gt_boxes"
      - "gt_classes"
      - "gt_masks"
      - "gt_keypoints"
    - "sem_seg": Tensor[int] in (H, W)
    - "proposals"
  - output: dict
    - "instances"
      - "pred_boxes"
      - "scores"
      - "pred_classes"
      - "pred_masks"
      - "pred_keypoints"
    - "sem_seg": Tensor in (num_categories, H, W)
    - "proposals"
      - "proposal_boxes"
      - "objectness_logits"
    - "panoptic_seg": tuple of (pred, segments_info)
      - pred
      - segments_info: List[Dict]
        - "id"
        - "isthing"
        - "category_id"



## Configuration systems

- lazy configs
  - https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
  - preferred
- yacs configs
  - https://detectron2.readthedocs.io/en/latest/tutorials/configs.html
  - legacy