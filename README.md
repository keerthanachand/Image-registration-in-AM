# Image Registration in Additive Manufacturing

This repository contains scripts for **XCT data preprocessing, machine learning-based data preparation, model training & testing, and data visualization**.  
The **primary goal** of this project is actaul nomimal comparison of **X-ray Computed Tomography (XCT) with Computer-Aided Design (CAD)** models to analyze **deformation in 3D-printed AM structures**.  

## 📂 Project Workflow  

### 1️⃣ **Linear Registration of XCT and CAD**  
- **Intensity-based or Point Cloud-based Registration**:  
  - Align XCT and CAD models using traditional registration methods.  
  - Compute transformation matrices for initial alignment.  

### 2️⃣ **Data Preparation for Deep Learning**  
 
- **Preprocessing XCT Data**:  
  - Remove powder artifacts and close pores to enhance XCT quality.  
  - Normalize intensity values to match CAD features.  
- **Convert Data into HDF5 Format**:  
  - Structure datasets for deep learning model training.  
  - Store voxel-based data efficiently.  

### 3️⃣ **Training the VoxelMorph Model**  
- **Train on Preprocessed XCT and CAD Data**:  
  - Implement an unsupervised deep learning approach using VoxelMorph.  
  - Optimize model performance for deformation prediction.  

### 4️⃣ **Testing & Quantification of Results**  
- **Apply Model to New XCT Scans**: Generate deformation fields.  
- **Evaluate Performance**:  
  - Compare model-predicted deformations with known deformations.  
  - Calculate voxel-wise displacement metrics.  

### 5️⃣ **Visualization & Analysis**  
- **Generate Displacement Fields**:  
  - Visualize deformations using 3D vector field plots.  
- **Analyze Differences Between XCT and CAD**:  
  - Identify deformation zones and quantify material shrinkage/expansion.  

---

## 🛠 Technologies Used  
- **Python** (NumPy, SciPy, OpenCV, TensorFlow/PyTorch)  
- **Deep Learning** (VoxelMorph, TensorFlow, Keras)  
- **3D Data Processing** (SimpleITK, Open3D, PyVista)  
- **Visualization** (Matplotlib, Seaborn, Plotly)  
- **HDF5 for Data Storage**  

--- 
## 📂 Directory Structure  

/deform_reg_project │── registration/ # XCT-CAD registration scripts │── preprocessing/ # Data preprocessing scripts │── ml_data_preparation/ # Data preparation for DL training │── model_training/ # VoxelMorph training scripts │── testing/ # Model testing and evaluation scripts │── visualization/ # Deformation visualization scripts │── requirements.txt # Dependencies list │── README.md # Project documentation

## 🚀 Installation
Clone this repository:

```bash
git clone https://github.com/keerthanachand/Image-registration-in-AM.git
cd Image-registration-in-AM

Install Dependencies

pip install -r requirements.txt