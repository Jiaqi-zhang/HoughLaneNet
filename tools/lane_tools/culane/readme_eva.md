# CULane Evaluation

## Unofficial Metric Implementation

https://github.com/lucastabelini/LaneATT

culane_metric.py: 
Unofficial implementation of the CULane metric. This implementation is faster than the oficial, however, it does not matches exactly the results of the official one (error in the order of 1e-4). Thus, it was used only during the model's development. For the results reported in the paper, the official one was used.

## Official Metric Implementation

https://xingangpan.github.io/projects/CULane.html

1. To evaluate your method, you may use evaluation code in this repo.
   https://github.com/XingangPan/SCNN
   
2. To generate per-pixel labels from raw annotation files, you could use this code.
    https://github.com/XingangPan/seg_label_generate
   
