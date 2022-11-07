# Detection Detailer
An object detection and auto-mask extension for [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). See [Installation](https://github.com/dustysys/ddetailer/README.md#installation).

![adoringfan](/misc/ddetailer_example_1.png)

### Segmentation
Default models enable person and face instance segmentation.

![amgothic](/misc/ddetailer_example_2.png)

### Detailing
With full-resolution inpainting, the extension is handy for improving faces without the hassle of manual masking.

![zion](/misc/ddetailer_example_3.gif)

## Installation
1. Use `git clone https://github.com/dustysys/ddetailer.git` from your SD web UI `/extensions` folder.
2. Start or reload SD web UI.

The models and dependencies should download automatically. To install them manually, follow the [official instructions for installing mmdet](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-mim-recommended). The models can be [downloaded here](https://huggingface.co/dustysys/ddetailer) and should be placed in `/models/mmdet/bbox` for bounding box (`anime-face_yolov3`) or `/models/mmdet/segm` for instance segmentation models (`dd-person_mask2former`). See the [MMDetection docs](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html) for guidance on training your own models.

## Usage
Select Detection Detailer as the script in SD web UI to use the extension. Here are some tips:
- `anime-face_yolov3` can detect the bounding box of faces as the primary model while `dd-person_mask2former` isolates the head's silhouette as the secondary model by using the bitwise AND option.
- The dilation factor expands the mask, while the x & y offsets move the mask around.
- The script is available in txt2img mode as well and can improve the quality of your 10 pulls with moderate settings.

## Credits
hysts/[anime-face-detector](https://github.com/hysts/anime-face-detector) - Creater of `anime-face_yolov3`, which has impressive performance on a variety of art styles.

skytnt/[anime-seg](https://huggingface.co/skytnt/anime-seg) - Synthetic dataset used to train `dd-person_mask2former`.

jerryli27/[AniSeg](https://github.com/jerryli27/AniSeg) - Annotated dataset used to train `dd-person_mask2former`.

open-mmlab/[mmdetection](https://github.com/open-mmlab/mmdetection) - Object detection toolset. `dd-person_mask2former` was trained via transfer learning using their [R-50 Mask2Former instance segmentation model](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former#instance-segmentation) as a base.

AUTOMATIC1111/[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Web UI for Stable Diffusion, base application for this extension.
