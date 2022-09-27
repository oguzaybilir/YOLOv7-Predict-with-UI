

# YOLOv7 Image and Video Inference with UI

This project is made for having quick inferences and makes the job easier for end users.
We aimed to develop a light project that makes fast inferences.

As an example, we used pothole dataset. 

The system features:

    OS : Ubuntu 20.04 LTS 64-bit 
    CPU : Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz
    GPU : Nvidia GeForce GTX 1650ti 4GB
    RAM : Samsung M471A1K43DB1-CWE 8GB

As you can see above, this project can also work on mid-segment systems.

## Inferences on Photo and Video

![into gif](https://github.com/oguzaybilir/YOLOv7-Predict-with-UI/blob/main/gif/fotograf.gif)

![into gif](https://github.com/oguzaybilir/YOLOv7-Predict-with-UI/blob/main/gif/video.gif)



## Cloning the Repository

Clone this repository with git.

```bash
  git clone https://github.com/oguzaybilir/YOLOv7-Predict-with-UI.git
  cd YOLOv7-Predict-with-UI
```
## Installing Libs

There is a requirements.txt file to install packages you need. This file contains almost all libraries and modules used in the project.

To install this libraries and packages:

```bash
    pip3 install -r requirements.txt
```
## Required Packages

These packages are absolutely essential packages for this project. In this case, you must first install the following packages in this order.

The *nvidia-driver-xxx* is your driver which is compatible with your graphic card.

        nvidia-driver-xxx
        CUDA == 11.6.2
        torch == 1.12.1
        TensorRT == 8.4
        
        

## TRT Weights and Sources

The converted trained weights and sources are stored in the drive link below.

[Drive Link](https://drive.google.com/drive/folders/15hrCM2OF30o5S4bpa5fD8nrVFm4aDjht?usp=sharing) to .trt weights and sources.


## Converting .PT weights to .ONNX and .TRT weights

**Method 1**

*Pytorch to ONNX with NMS (and inference)*  -  [Open In Colab](https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb)
```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Method 2**

*Pytorch to TensorRT with NMS (and inference)* -  [Open In Colab](https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb)

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640

git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Method 3**

*Pytorch to TensorRT another way*  - [Open In Colab](https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb)


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms

git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

## Run 
```bash
  python3 main.py
```

## Authors

- [@oguzaybilir](https://github.com/oguzaybilir)
- [@furkantahabademci](https://github.com/furkantahabademci)



## Acknowledgements

 - [Mehmet Okuyar](https://github.com/MehmetOKUYAR)
 - [Official YOLOv7 Repository](https://github.com/WongKinYiu/yolov7)
 - [TensorRT Website](https://github.com/matiassingers/awesome-readme)
 - [CUDA Website](https://developer.nvidia.com/cuda-zone)
