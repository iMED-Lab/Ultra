<p align="center">   
    <p align="center">
    <img src="https://capsule-render.vercel.app/api?type=transparent&height=150&text={%20Ultra%20}&reversal=false&section=header&textBg=false&fontColor=E90064&fontSize=40&animation=scaleIn&desc=Multi-Granularity%20Topological%20Reasoning%20for%20Anatomically%20Consistent%20Vasculature%20Parsing&descSize=20&descAlignY=80" />
   <img alt="OS - Ubuntu" src="https://img.shields.io/badge/OS-Ubuntu-E95420?logo=ubuntu&logoColor=white"/>
  <img alt="Python - 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white"/>
  <img alt="PyTorch - 2.2.0+" src="https://img.shields.io/badge/PyTorch-2.2.0%2B-EE4C2C?logo=pytorch&logoColor=white"/>
  <img alt="nnUNetv2" src="https://img.shields.io/badge/nnUNet-v2-005CED?logo=onnx&logoColor=white"/>
  <img alt="CUDA 12.8" src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white"/>
  <img alt="Env - Anaconda3" src="https://img.shields.io/badge/Env-Anaconda3-44A833?logo=anaconda&logoColor=white"/>
</p>
<p align="center">
 <img width="800px" src="https://i.postimg.cc/14hrF3y4/segav.png" align="center" alt="Segmentation results"/>
</p>
---

## :rocket: Installation 

### 1. Clone the Repository

Download the repository and navigate to the project directory: 

``````
git clone https://github.com/iMED-Lab/Ultra.git
cd Ultra
``````

 ### 2. Create a Conda Environment

It is recommended to create and activate a dedicated Conda environment for the framework: 

```
conda create -n ultra python=3.11 -y
conda activate ultra
```

### 3. Install the Ultra Framework

Install the required dependencies and the framework itself: 

``````
pip install -e .
``````

---

## :traffic_light: Usage

> **Note:** Make sure to configure your nnUNet paths as described in the [official instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) before executing these commands.

### Training

To train the model, specify the dataset ID, configuration, and fold. 

```
ultra-train <dataset_id> <configuration> <fold>
```

*Example:* `ultra-train 888 2d 0`

### Inference

To generate Artery/Vein (A/V) predictions using your trained models, use the following command:

```
ultra-predict -i <input_folder> -o <output_folder> -d <dataset_id> -c <configuration|default=2d> -f <fold>
```

> **Note:** The checkpoint with the best performance during training will be automatically selected as the default for prediction.

### Neighborhood Connectivity Encoding

The core implementation for our neighborhood connectivity encoding is located in [`ultra/utilities/to_neighbor_connectivity.py`](ultra/utilities/to_neighbor_connectivity.py). This module captures multi-scale spatial relationships to preserve structural continuity.

**Key Components:** 

* **`to_nk_maps` (Entry Point):** Aggregates neighborhood connectivity maps across multiple receptive fields (e.g., kernel sizes of 3, 5, and 7).  
* **`nk_encode`:** Computes the connectivity encoding for a specific kernel size $k$. It evaluates the structural continuity between a center pixel and its surrounding neighborhood. 
* **`bresenham_line`:** An underlying utility that implements Bresenham's line algorithm to accurately determine the discrete pixel path between two points during the connectivity check.

### Trained Models

Pre-trained model weights are available for download on [CFP-AV](https://doi.org/10.5281/zenodo.18939161), [UWF-AV](https://doi.org/10.5281/zenodo.18939161), [CFP+UWF-AV](https://doi.org/10.5281/zenodo.18973075). To perform inference, please download and place the model folders into the directory specified by your `nnUNet_results` environment variable (refer to the [nnUNet instructions for setting up paths](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) for more details).

## :bookmark: Citation

*Our paper is currently under review. Citation details are coming soon!*

---

## :heart: Acknowledgements

We would like to extend our deepest gratitude to the following contributors and organizations whose support has been instrumental in the development of the **Ultra** framework: 

- The brilliant developers and maintainers of [**nnU-Net v2**](https://github.com/MIC-DKFZ/nnUNet), whose robust segmentation framework heavily inspired this project. 
- The broader research community in retinal imaging and medical image analysis for continuously providing valuable insights and high-quality benchmark datasets.
