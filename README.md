# Image Registration in Additive Manufacturing

This repository contains Python scripts for image registration and deformation analysis of additive manufactured parts using X-ray Computed Tomography (XCT) and Computer-Aided Design (CAD) data.

The main goal of the project is to perform actual-to-nominal comparison between XCT scans and CAD models and to analyze geometric deviations in 3D-printed structures.

## Workflow

### 1. XCT–CAD Registration

XCT and CAD data are first aligned using linear registration methods such as intensity-based registration or point-cloud-based registration.

### 2. Data Preprocessing

The XCT data are preprocessed before deep learning-based registration. This includes steps such as:

- removing powder artifacts
- filling pores
- thresholding and normalization
- preparing paired XCT and CAD volumes
- storing the processed data in HDF5 format

### 3. VoxelMorph Training

VoxelMorph is used for non-linear image registration between XCT and CAD data.

The repository includes scripts for:

- model training
- cross-validation
- fine-tuning
- hyperparameter optimization
- ensemble training

### 4. Evaluation

The trained models are evaluated using registration and deformation metrics such as:

- Dice coefficient
- voxel-wise difference maps
- displacement fields

The registered volumes and displacement fields can also be exported for further analysis.

### 5. Deformation Analysis and Compensation

The predicted displacement fields are used to analyze geometric deviations between the manufactured part and the nominal CAD model.

The repository also contains scripts for applying the predicted deformation to mesh data for geometry compensation.

## Main Technologies

- Python
- TensorFlow / Keras
- VoxelMorph
- NumPy
- SciPy
- scikit-image
- HDF5
- PyVista
- Matplotlib

## Directory Structure

```text
Image-registration-in-AM/
├── notebooks/
├── src/
│   ├── compensation/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── scripts/
│   ├── training/
│   ├── visualization/
│   └── results/
├── requirements.txt
├── .gitignore
└── README.md