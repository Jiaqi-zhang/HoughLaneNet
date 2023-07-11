# Datasets
We conducted experiments on TuSimple, CULane and LLAMAS. The settings of these datasets are as follows. 
Note, [your-data-path] is the path you specifiying to save the datasets and [project-root] is the root path of this project.


## TuSimple
[\[Website\]](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)

Inside [your path], run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
# train & validation data (~10 GB) & test images (~10 GB) & test annotations
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/dhbloo/TuSimple
cd ..
```

Then the directory should be like follows:
```
[your-data-path]/TuSimple
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── test_label.json
└── test_baseline.json

```


## CULane
[\[Website\]](https://xingangpan.github.io/projects/CULane.html)

Inside [your-data-path], run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
mkdir culane
# train & validation images (~30 GB)
gdown "https://drive.google.com/uc?id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk"
gdown "https://drive.google.com/uc?id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL"
gdown "https://drive.google.com/uc?id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL"
tar xf driver_23_30frame.tar.gz
tar xf driver_161_90frame.tar.gz
tar xf driver_182_30frame.tar.gz
# test images (~10 GB)
gdown "https://drive.google.com/uc?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8"
gdown "https://drive.google.com/uc?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"
gdown "https://drive.google.com/uc?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7"
tar xf driver_37_30frame.tar.gz
tar xf driver_100_30frame.tar.gz
tar xf driver_193_90frame.tar.gz
# all annotations (train, val and test)
gdown "https://drive.google.com/uc?id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu"
tar xf annotations_new.tar.gz
gdown "https://drive.google.com/uc?id=1R9hSJQIof3q66JzV3Gte7q6WQ_TqZCDN"
tar xf list.tar.gz
gdown "https://drive.google.com/uc?id=1k5FkMt9QKCuWT1_nJcmD3i7iiP8SvXAx"
tar xf laneseg_label_w16.tar.gz

# We convert to TuSimple formal using prepare_culane_datasets.py script
cd [project-root]
python tools/lane_tools/culane/prepare_culane_datasets.py [your-data-path/CULane]
```

Then the directory should be like follows:
```
[your-data-path]/CULane
├── driver_23_30frame
├── driver_37_30frame
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
├── json_lanes_train
├── json_lanes_test
├── json_lanes_val
├── laneseg_label_w16
└── list
    └── test_split
    |   ├── test0_normal.txt
    |   ├── test1_crowd.txt
    |   ├── test2_hlight.txt
    |   ├── test3_shadow.txt
    |   ├── test4_noline.txt
    |   ├── test5_arrow.txt
    |   ├── test6_curve.txt
    |   ├── test7_cross.txt
    |   └── test8_night.txt
    └── train.txt
    └── train_gt.txt
    └── val.txt
    └── val_gt.txt
    └── test.txt

```




## LLAMAS
[\[Website\]](https://unsupervised-llamas.com/llamas/)

```bash
mkdir datasets # if it does not already exists
cd datasets

# Please refer to the official website for download instructions, and download the required data.

# We convert to TuSimple formal using prepare_llamas_datasets.py script
cd [project-root]
python tools/lane_tools/llamas/prepare_llamas_datasets.py [your-data-path/LLAMAS]
```

Then the directory should be like follows:
```
[your-data-path]/LLAMAS
├── json_lanes_train
├── json_lanes_valid
├── train.txt
├── valid.txt
├── color_images
|   ├── train
|   ├── valid
|   └── test
└── labels
    └── train
    └── valid

```

