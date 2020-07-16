![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# Interactive Video Object Segmentation Using Global and Local Transfer Modules
#### Yuk Heo, Yeong Jun Koh, Chang-Su Kim

![IVOS Image](Overall_Network.png)

[Project page](https://openreview.net/forum?id=bo_lWt_aA)

Pytorch implementation of [ECCV2020 paper](https://openreview.net/forum?id=bo_lWt_aA) in this page:

1. DAVIS2017 evaluation based on the [DAVIS framework](https://interactive.davischallenge.org/)
2. DAVIS2016 real-world evaluation GUI

### Prerequisite
- python 3.6
- pytorch 1.2.0
- [davisinteractive 1.0.4](https://github.com/albertomontesg/davis-interactive)
- [corrlation package](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package) of [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)
- numpy, cv2, PtQt5, and other general libraries of python3

### Directory Structure
 * `ROOT/libs`: library of utility files.

 * `ROOT/networks` : network codes
     - `correlation_package` : conserves GPU memory by appling the correlation package of FlowNet2
     - `deeplab`: applies ASPP module in decoders, [[original code]](https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master/modeling)
     - `atnet.py`: consists A-Net and T-Net
     - `ltm_transfer.py`: transfers previous segmentation with the local affinity of the local transfer module

 * `ROOT/config`  : configurations
 
 * `ROOT/eval_davisframework` : DAVIS2017 evaluation based on the [DAVIS framework](https://interactive.davischallenge.org/)
 
 * `ROOT/eval_real-world` : DAVIS2016 real-world evaluation GUI, (to be opened)

### Instruction



### Reference

Please cite our paper if the implementations are useful in your work:
```
@Inproceedings{
Yuk2020IVOS,
title={Interactive Video Object Segmentation Using Global and Local Transfer Modules},
author={Yuk Heo and Yeong Jun Koh and Chang-Su Kim},
booktitle={ECCV},
year={2020},
url={https://openreview.net/forum?id=bo_lWt_aA}
}
```

Our real-world evaluation demo is based on the GUI of [IPNet](https://github.com/seoungwugoh/ivs-demo):
``` 
@Inproceedings{
Oh2019IVOS,
title={Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks},
author={Seoung Wug Oh and Joon-Young Lee and Seon Joo Kim},
booktitle={CVPR},
year={2019},
url={https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.pdf}
}
```
