# 🧠 AI-Powered Glioma Segmentation & Classification with Vision Transformers 🚀


This project tackles the critical task of **detecting, segmenting, and classifying gliomas (brain tumors)** from multi-modal Magnetic Resonance Imaging (MRI) scans. We leverage state-of-the-art deep learning techniques, primarily focusing on **Vision Transformers (ViTs)** and **hybrid CNN-Transformer architectures**, to achieve accurate and robust results on the challenging BraTS 2020 dataset.

➡️ **View the Detailed Project Report [PDF](https://drive.google.com/file/d/10jQ3m5n0LZsOeFTjLLVYp0X_we7kIw_l/view?usp=sharing)** ⬅️
---

➡️ **View the Presentation here [Presentation](https://docs.google.com/presentation/d/1HHhFd5_X9-WG9Crv0i_Z5zmyrCdi9gOY/edit?usp=sharing&ouid=112907639384456184334&rtpof=true&sd=true)** ⬅️
---

## 🌟 Key Features

*   **Multi-Modal MRI Processing:** Handles and fuses information from various MRI sequences (T1, T1Gd, T2, T2-FLAIR).
*   **Advanced Segmentation Models:**
    *   Implementation and fine-tuning of **BEFUnet**, a novel hybrid CNN-Transformer architecture, achieving **0.805 mIoU** on BraTS 2020.
    *   Comparative analysis with other SOTA models: **SegFormer, Swin-UNet, MobileViTV2**.
*   **Glioma Grade Classification:** A separate deep learning model (CNN-based) to classify the type/grade of the segmented tumor.
*   **End-to-End Pipeline:** Comprehensive workflow from data preprocessing, augmentation, model training, and evaluation to prediction.
*   **In-depth Evaluation:** Utilizes standard metrics like Mean Intersection over Union (mIoU), Dice Coefficient, Precision, Recall, and F1-Score.
*   **Interactive UI Prototype:** A Flask-based web application for visualizing model predictions on new MRI slices.

---

## 🛠️ Technical Stack

*   **Programming Language:** Python 3.x
*   **Deep Learning Frameworks:** PyTorch, TensorFlow/Keras
*   **Core Libraries:**
    *   **Vision & Transformers:** Hugging Face Transformers
    *   **Data Handling:** Pandas, NumPy, H5py, Nibabel
    *   **Image Processing:** OpenCV, Scikit-image
    *   **Machine Learning:** Scikit-learn
    *   **Plotting:** Matplotlib, Seaborn
*   **Web Framework (UI Prototype):** Flask
*   **Development Environment:** Jupyter Notebooks, Kaggle Kernels

---

## 📊 Models & Methodology

This project explores and implements several advanced architectures for brain tumor analysis:

1.  **BEFUnet (Our Primary Segmentation Model):**
    *   A hybrid architecture combining a **CNN branch (PDC-based)** for edge feature extraction and a **Swin Transformer branch** for capturing global body context.
    *   Utilizes a **Double-Level Fusion (DLF)** module and **Local Cross-Attention Feature Fusion (LCAF)** for effective merging of multi-scale features.
    *   Trained using a combined Dice Loss and Cross-Entropy Loss.

2.  **Comparative Segmentation Models:**
    *   **SegFormer:** Transformer-based model with a hierarchical encoder and lightweight MLP decoder.
    *   **SegFormer-UNet:** Hybrid model combining SegFormer as an encoder with a UNet decoder.
    *   **Swin-UNet:** Pure Transformer U-Net architecture using Swin Transformer blocks with shifted windows.
    *   **MobileViTV2:** Efficient hybrid Vision Transformer for mobile applications, adapted for segmentation.

3.  **Glioma Classification Model:**
    *   A CNN-based model (using Keras) trained on segmented tumor regions to classify glioma grades (e.g., Glioma, Meningioma, No tumor, Pituitary - adjust based on your classification task).

### Data Preprocessing & Augmentation:
*   **Segmentation:** Resizing, random flips, rotations, zoom, channel fusion (e.g., max across modalities), normalization.
*   **Classification:** Resizing, channel combination from segmented outputs.

---

## 🏆 Key Results

*   **BEFUnet Segmentation Performance (BraTS 2020):**
    *   **Mean Intersection over Union (mIoU): 0.805**
    *   Average Dice Coefficient: 0.154 (*Note: Dice Loss is `1 - Dice Coef`, so high loss means low coef. The report shows Dice_Coef, which might be `1 - Dice_Loss`.* Ensure this metric is presented correctly. A Dice Coef of 0.154 is low; 0.846 would be high. Clarify if `avg_dice_coef` in code is Dice Score or Dice Loss)
    *   Average Precision: 0.393
    *   Average Recall: 0.335
    *   Average F1-Score: 0.353

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Anaconda or Miniconda (recommended for managing environments)
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BinarySkull/BEFUnet-MRI-Segmentation.git
    BEFUnet-MRI-Segmentation
    ```

2.  **Create and activate a conda environment (optional but recommended):**
    ```bash
    conda create -n glioma-env python=3.9
    conda activate glioma-env
    ```

3.  **Install dependencies:**
    Using pip or conda
    
5.  **Download Datasets:**
    *   **BraTS 2020 Training Data:** Download from [The BraTS Challenge website](https://www.med.upenn.edu/cbica/brats2020/registration.html) (requires registration). Place the data in a directory structure expected by the notebooks (e.g., `/kaggle/input/brats2020-training-data/...` or update paths in the code).

6.  **Model Weights & Dependencies:**
    The main notebook might clone external repositories for specific model architectures or pre-trained weights.
    *   The `BEFUnet_Brats2020` model and weights are sourced from `https://huggingface.co/Unknown6197/BEFUnet_Brats2020`. The code automatically clones this.
    *   The classification model resources are from `https://huggingface.co/Unknown6197/res_classification`. The code automatically clones this.


### Running the Code

1.  **Jupyter Notebook for Segmentation & Classification:**
    *   Open and run the main Jupyter Notebook.
    *   The notebook covers:
        *   Data loading and preprocessing.
        *   BEFUnet model definition, training (optional, can use pre-trained), and evaluation.
        *   Evaluation of other segmentation models (if cells are active).
        *   Prediction and visualization.
        *   Loading and using the classification model.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), Have Fun!

---

## 🙏 Acknowledgements

*   The BraTS 2020 Challenge organizers and data providers.
*   Authors of BEFUnet, SegFormer, Swin-UNet, MobileViTV2, and other foundational papers and repositories.
*   The Hugging Face team for their `transformers` library and model hub.

---

*Feel free to contribute, open issues, or suggest improvements!*
