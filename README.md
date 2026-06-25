<h2 align="center">RT-DETR++: DETRs Beat YOLOs on Real-time Object Detection</h2>
<p align="center">
    <img alt="python" src="https://img.shields.io/badge/Python-3.11+-blue">
    <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
    <img alt="task" src="https://img.shields.io/badge/Task-Real--Time%20Object%20Detection-brightgreen">
    <img alt="framework" src="https://img.shields.io/badge/Framework-RT--DETR++-orange">
</p>

---

Official codebase of **RT-DETR++**.

This work extends RT-DETR with stronger real-time performance, end-to-end deployment friendliness, and broader cross-domain generalization.

## ⚡ Performance

| Model | AP | AP50 | AP75 | FPS (T4) | Config |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RT-DETR++S | 49.8 | 66.9 | 54.1 | 253 | [yml](./configs/rtdetrplus/rtdetrplus_hgnetv2_s_coco.yml) |
| RT-DETR++M | 53.6 | 71.1 | 58.2 | 161 | [yml](./configs/rtdetrplus/rtdetrplus_hgnetv2_m_coco.yml) |
| RT-DETR++L | 55.5 | 73.1 | 60.4 | 119 | [yml](./configs/rtdetrplus/rtdetrplus_hgnetv2_l_coco.yml) |
| RT-DETR++X | 57.1 | 74.6 | 62.2 | 76 | [yml](./configs/rtdetrplus/rtdetrplus_hgnetv2_x_coco.yml) |

## Content

- [⚡ Performance](#-performance)
- [Content](#content)
- [1. Getting Started](#1-getting-started)
  - [Setup](#setup)
  - [Data Preparation](#data-preparation)
  - [Teacher Model Preparation (DINOv3)](#teacher-model-preparation-dinov3)
- [2. Usage](#2-usage)
- [3. Tools](#3-tools)

-----

## 1. Getting Started

### Setup

```shell
conda create -n rtdetrpp python=3.11.9
conda activate rtdetrpp
pip install -r requirements.txt
```

### Data Preparation

<details>
<summary> COCO2017 Dataset </summary>

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or [COCO](https://cocodataset.org/#download).
2. Modify dataset paths in [`configs/dataset/coco_detection.yml`](./configs/dataset/coco_detection.yml):

```yaml
train_dataloader:
    img_folder: /data/COCO2017/train2017/
    ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
    img_folder: /data/COCO2017/val2017/
    ann_file: /data/COCO2017/annotations/instances_val2017.json
```

</details>

<details>
<summary>Custom Dataset</summary>

To train on custom data, organize your dataset in COCO format.

1. Set `remap_mscoco_category: False`.
2. Update `num_classes`, `img_folder`, and `ann_file` in [`configs/dataset/custom_detection.yml`](./configs/dataset/custom_detection.yml).

</details>

### Teacher Model Preparation (DINOv3)

RT-DETR++ uses a VFM teacher model for distillation.

- DINOv3 repository: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Weights download: [DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

Edit `teacher_model` in `configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml`:

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
```

## 2. Usage

<details open>
<summary> COCO2017 </summary>

1. Training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

2. Evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml --test-only -r model.pth
```

3. Fine-tuning

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml --use-amp --seed=0 -t model.pth
```

</details>

<details>
<summary> Speed Tuning without Retraining </summary>

RT-DETR++ supports speed/accuracy trade-off by adjusting active decoder layers at inference time.
This is useful for deploy-anywhere scenarios with different latency budgets.

</details>

<details>
<summary> Customizing Batch Size </summary>

When changing `total_batch_size`, tune related hyperparameters in model config:

- `optimizer.lr`
- backbone LR in `optimizer.params`
- warmup settings (`warmup_iter` or warmup scheduler)
- EMA settings (if enabled)

</details>

<details>
<summary> Customizing Input Size </summary>

To switch input size (e.g., `320x320`), update:

- resize ops in [`configs/base/dataloader.yml`](./configs/base/dataloader.yml)
- `eval_spatial_size` in [`configs/base/rtdetrplus.yml`](./configs/base/rtdetrplus.yml)

</details>

## 3. Tools

<details>
<summary> Deployment </summary>

1. Setup

```shell
pip install onnx onnxsim
```

2. Export ONNX

```shell
python tools/deployment/export_onnx.py --check \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml -r model.pth
```

3. Export TensorRT engine

```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

</details>

<details>
<summary> Inference (Visualization) </summary>

1. Setup

```shell
pip install -r tools/inference/requirements.txt
```

2. Inference (onnxruntime / tensorrt / torch)

```shell
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg   # or video.mp4
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
python tools/inference/torch_inf.py \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```

</details>

<details>
<summary> Benchmark </summary>

1. Setup

```shell
pip install -r tools/benchmark/requirements.txt
```

2. FLOPs / MACs / Params

```shell
python tools/benchmark/get_info.py -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml
```

3. TensorRT latency benchmark

```shell
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

</details>

<details>
<summary> FiftyOne Visualization </summary>

```shell
pip install fiftyone
python tools/visualization/fiftyone_vis.py \
  -c configs/rtdetrplus/rtdetrplus_hgnetv2_${model}_coco.yml -r model.pth
```

</details>

<details>
<summary> Others </summary>

```shell
python tools/reference/convert_weight.py model.pth
python tools/dataset/remap_obj365.py
python tools/dataset/resize_obj365.py
```

</details>
