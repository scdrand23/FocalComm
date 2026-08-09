# FocalComm: Hard Instance-Aware Multi-Agent Perception

Official implementation of **FocalComm** (WACV 2026).

## Abstract

Multi-agent collaborative perception (CP) is a promising paradigm for improving autonomous driving safety, particularly for vulnerable road users like pedestrians, via robust 3D perception. However, existing CP approaches often optimize for vehicle detection performance metrics, underperforming on smaller, safety-critical objects such as pedestrians, where detection failures can be catastrophic. Furthermore, previous CP methods rely on full feature exchange rather than communicating only salient features that help reduce false negatives. To this end, we present FocalComm, a novel collaborative perception framework that focuses on exchanging hard-instance-oriented features among connected collaborative agents. FocalComm consists of two key novel designs: (1) a learnable progressive hard instance mining (HIM) module to extract hard instances-oriented features per agent, and (2) a query-based feature-level (intermediate) fusion technique that dynamically weights these identified features during collaboration.

## Installation

```bash
conda env create -f environment.yaml
conda activate focalcomm

python focalcomm/utils/setup.py build_ext --inplace
cd focalcomm/pcdet_utils && python setup.py build_ext --inplace
```

For H100/H200-class GPUs (CUDA 11.8 + PyTorch 2.0), use the alternative environment:

```bash
conda env create -f environment_h100.yaml
conda activate focalcomm_h100
```

## Dataset Preparation

### V2X-Real

Download from [V2X-Real](https://github.com/ucla-mobility/V2X-Real) and set the paths in `focalcomm/hypes_yaml/v2xreal/*.yaml`:

```yaml
root_dir: "/path/to/dataset/V2XReal/train"
validate_dir: "/path/to/dataset/V2XReal/val"
test_dir: "/path/to/dataset/V2XReal/test"
```

`dataset_mode` selects the collaboration setup: `vc` (vehicle-centric), `ic` (infrastructure-centric), `v2v`, or `i2i`.

### DAIR-V2X

Download the cooperative-vehicle-infrastructure split from [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) and set the paths in `focalcomm/hypes_yaml/dairv2x/*.yaml`:

```yaml
data_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure"
root_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/train.json"
validate_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/val.json"
test_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/val.json"
```

## Usage

### Training

```bash
python focalcomm/tools/train_focalcomm.py --hypes_yaml focalcomm/hypes_yaml/v2xreal/focalcommv3.yaml

python focalcomm/tools/train_focalcomm.py --hypes_yaml focalcomm/hypes_yaml/dairv2x/focalcomm.yaml
```

### Inference

```bash
python focalcomm/tools/inference.py --model_dir checkpoints/focalcomm_v2xreal --fusion_method intermediate

python focalcomm/tools/inference.py --model_dir checkpoints/focalcomm_dairv2x --fusion_method intermediate
```

`--model_dir` accepts any directory containing a `config.yaml` and `net_epoch*.pth`; the latest epoch is loaded by default (`--epoch` overrides).

## Results

Performance on V2X-Real (Vehicle-Centric / Infrastructure-Centric) and DAIR-V2X, reported as AP@0.3/AP@0.5 (from the paper):

| Method | Car (VC) | Car (IC) | Pedestrian (VC) | Pedestrian (IC) | Truck (VC) | Truck (IC) | Overall (VC) | Overall (IC) | DAIR-V2X |
|--------|----------|----------|-----------------|-----------------|------------|------------|--------------|--------------|----------|
| F-Cooper | 88.3/85.6 | 84.3/80.8 | 47.8/22.7 | 45.4/15.9 | 47.9/46.1 | 48.3/47.9 | 61.3/51.4 | 59.4/48.2 | 70.4/64.8 |
| V2VNet | 87.0/84.4 | 85.0/81.4 | 34.5/13.9 | 36.5/15.2 | 40.0/36.8 | 44.3/41.9 | 53.8/45.0 | 55.3/46.2 | 69.5/63.5 |
| AttFuse | 81.3/80.7 | 81.5/80.9 | 46.8/21.7 | 48.5/24.8 | 49.6/47.7 | 47.6/45.7 | 59.2/50.0 | 59.2/50.5 | 69.7/63.8 |
| CoBEVT | 87.2/85.6 | 84.1/82.1 | 54.8/26.1 | 52.3/25.6 | 50.1/45.1 | 48.9/47.8 | 64.0/53.3 | 61.7/52.9 | 72.8/65.7 |
| V2X-ViT | 83.9/81.1 | 81.4/78.2 | 38.5/15.2 | 33.5/13.3 | 42.5/35.6 | 45.4/38.9 | 55.0/44.0 | 53.4/43.5 | 74.5/67.6 |
| CoAlign | 85.8/83.4 | 84.7/83.4 | 38.3/17.3 | 36.4/14.8 | 52.7/43.9 | 53.2/51.1 | 59.9/48.2 | 58.1/49.8 | 76.9/69.7 |
| ERMVP | 88.5/86.4 | 86.7/84.0 | 53.2/25.4 | 50.6/23.5 | 42.9/41.3 | 41.7/38.7 | 61.5/51.0 | 59.7/48.7 | 69.2/63.4 |
| **FocalComm (ours)** | **91.5/89.6** | 86.2/**84.8** | **57.4/27.3** | **51.2**/26.7 | **53.9/51.6** | **49.6**/47.3 | **67.6/56.1** | **62.3**/52.9 | 73.3/66.4 |

V2V and I2I communication scenarios (Overall mAP@0.3/mAP@0.5): FocalComm reaches **63.3/49.6** (V2V) and **68.8/56.0** (I2I).

## Pretrained Checkpoints

Pretrained weights are hosted on Hugging Face at [scdrand23/FocalComm](https://huggingface.co/scdrand23/FocalComm). Download them into `checkpoints/`:

```bash
pip install -U huggingface_hub
hf download scdrand23/FocalComm --local-dir checkpoints
```

Each checkpoint ships with its exact training `config.yaml`:

| Checkpoint | Dataset | Config | Measured AP@0.3/AP@0.5 |
|------------|---------|--------|------------------------|
| `checkpoints/focalcomm_v2xreal` | V2X-Real (VC) | FocalComm 3-stage HIM, epoch 50 | Car 91.6/88.4, Ped 53.2/26.1, Truck 50.4/47.2, Overall 65.1/53.9 |
| `checkpoints/focalcomm_dairv2x` | DAIR-V2X | FocalComm, epoch 50 | Vehicle 73.3/66.4 |

The DAIR-V2X checkpoint reproduces the paper result exactly. The V2X-Real checkpoint is the final epoch of the paper's training run; its stored evaluation is within ~2 mAP of the paper's Table 1 row (the exact epoch used for the paper table was not preserved).

Before running inference, update the dataset paths in each checkpoint's `config.yaml` to point to your local dataset.

## Citation

```bibtex
@inproceedings{shenkut2026focalcomm,
  title={FocalComm: Hard Instance-Aware Multi-Agent Perception},
  author={Shenkut, Dereje and Bhagavatula, Vijayakumar},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## Acknowledgments

This codebase builds on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), and the [V2X-Real](https://github.com/ucla-mobility/V2X-Real) benchmark.

## License

MIT License
