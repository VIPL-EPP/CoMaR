# Collaborative Map-based and Route-based Policy Learning for Continuous Vision-and-Language Navigation

This repository is the official implementation of: Collaborative Map-based and Route-based Policy Learning for Continuous Vision-and-Language Navigation

> Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow language instructions to reach targets in unseen 3D environments, demanding both spatial reasoning and procedural alignment during cross-modal planning. Existing methods typically emphasize only one of these abilities: map-based policies support spatial reasoning via graph representations, while route-based policies favor procedural alignment by matching sequential observations with instructions. Studied in isolation, such policies limit effective planning in complex scenes. Inspired by human navigation, we propose a collaborative policy learning framework that integrates both capabilities. Our framework comprises Spatio-Procedural Topological Mapping to build a multiplex graph, Dual-Stream Encoding for parallel cross-modal reasoning, and Hierarchical Policy Integration that fuses the two policies at feature and logit levels. Extensive experiments on VLN-CE benchmarks demonstrate the effectiveness of our approach.

![Model Architecture](img/Model.jpg)

## TODO

- [x] Release evaluation model checkpoints and evaluation code for CoMaR<sub>ETPNav</sub>.
- [x] Release evaluation model checkpoints and evaluation code for CoMaR<sub>g3D-LF</sub>.
- [ ] Release training pipeline and training code for CoMaR<sub>ETPNav</sub>.
- [ ] Release training pipeline and training code for CoMaR<sub>g3D-LF</sub>.
- [ ] Release real-world deployment code on the Unitree robot dog.


## Requirements

To set up the conda environment, please follow the guidelines provided by the baseline methods: [ETPNav](https://github.com/MarSaKi/ETPNav) and [g3D-LF](https://github.com/MrZihan/g3D-LF). Configuring the environment for these baselines is sufficient to meet all the requirements for CoMaR; no additional dependencies are needed. For your convenience, we also provide a ready-to-use environment configuration file: `env.yml`.Specifically, the environment setup process of the baselines is as follows:

#### 1. Install Habitat Simulator:
Follow the installation instructions from [ETPNav](https://github.com/MarSaKi/ETPNav) or the original [VLN-CE](https://github.com/jacobkrantz/VLN-CE) repository.

#### 2. Install `torch_kdtree` (Required for g3D-LF):
For K-nearest feature search in g3D-LF, install `torch_kdtree` from the [official repository](https://github.com/thomgrand/torch_kdtree):
```bash
git clone [https://github.com/thomgrand/torch_kdtree](https://github.com/thomgrand/torch_kdtree)
cd torch_kdtree
git submodule init
git submodule update
pip3 install .

```

#### 3. Install `tinycudann` (Required for g3D-LF):
For faster multi-layer perceptrons (MLPs) in g3D-LF, install `tinycudann` from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):

```bash
pip3 install git+[https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch](https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch)

```

#### 4. Download Checkpoints
Download the required model checkpoints from our [Google Drive](https://drive.google.com/drive/folders/1WDvmWeZ6c4uPqaGEGufvSfigFh7che4u?usp=drive_link).


## Evaluation

We have adapted the CoMaR framework for the baseline methods [ETPNav](https://github.com/MarSaKi/ETPNav) and [g3D-LF](https://github.com/MrZihan/g3D-LF). The specific implementations are included in the new `vlnce_baselines` folder provided in this repository. 

To evaluate the models, simply replace the original `vlnce_baselines` folder in the respective baseline repositories with the one we provide. The overall evaluation process follows the original guidelines from ETPNav and g3D-LF.Specifically, you can run the following commands for evaluation:

### For ETPNav

```bash
# Evaluate on R2R-CE
CUDA_VISIBLE_DEVICES=0,1 bash run_r2r/main.bash eval 2333

# Evaluate on RxR-CE
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main.bash eval 2333
```

### For g3D-LF

```bash
# Evaluate on R2R-CE
bash run_r2r/main.bash eval 2347

# Evaluate on RxR-CE
bash run_rxr/main.bash eval 2347
```

## Acknowledgements
Our code is based on [g3D-LF](https://github.com/MrZihan/g3D-LF), [PRET](https://github.com/iSEE-Laboratory/VLN-PRET) and [ETPNav](https://github.com/MarSaKi/ETPNav). Thanks for their great works!