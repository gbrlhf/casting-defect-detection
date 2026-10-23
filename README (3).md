# 🔍 Casting Defect Detection

**AI-based visual quality inspection for metal casting products — classifying parts as OK or Defective, with explainable results for QC operators.**

> 🚧 **Status: In progress.** This README describes the problem and the planned solution. Results will be added once the model is trained and evaluated.

---

## 📌 Background

Metal casting is a core process in automotive component manufacturing, from engine parts to **aluminum alloy wheels**, which are commonly produced by *low pressure die casting*. During casting, molten metal can solidify imperfectly and leave defects such as:

- **Porosity / blow holes**: gas trapped inside the metal
- **Shrinkage**: cavities caused by uneven cooling
- **Cracks and pinholes**
- **Surface irregularities**: burrs, flash, or rough surfaces

A defective part that reaches the customer can lead to **product failure, warranty claims, and rejected shipments**. For safety-critical parts like wheels, this risk is unacceptable.

## ❗ Problem Statement

In many factories, final inspection is still done **manually by human operators**, a procedure that research describes as time-consuming and error-prone [[1]](#-references), with several limitations:

| Problem | Impact |
|---|---|
| **Slow** compared to production speed | Inspection becomes a bottleneck on high-volume lines |
| **Subjective**: results vary between operators and shifts | Inconsistent quality standards |
| **Fatigue**: accuracy drops over long shifts | Defective parts slip through (false negatives) |
| **No data trail**: decisions are not recorded systematically | Hard to analyze defect trends and root causes |

Automated visual inspection with deep learning has become a widely studied alternative for industrial quality control [[2]](#-references), including for aluminum castings in the automotive industry [[3]](#-references) [[4]](#-references).

**Goal:** build a model that can automatically and consistently classify casting products as **OK** or **Defective** from images, and explain *why* it made each decision, so that it can support (not replace) human inspectors.

---

## 💡 Proposed Solution

```
 Product image ──► Preprocessing ──► CNN Classifier ──► OK / Defective + confidence
                                          │
                                          └──► Grad-CAM heatmap (where the defect is)
                                                     │
                                                     ▼
                                    Web demo for QC operator + inspection log
```

### 1. Dataset
Using the public **[Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)** (Kaggle), which contains top-view grayscale images of cast **submersible pump impellers**, labeled as [[5]](#-references):
- `ok_front`: good product
- `def_front`: defective product

> ⚠️ **Note:** the images are pump impellers, not wheels. They are used here because they show **real casting defects** (porosity, blow holes, shrinkage) of the same kind found in cast aluminum wheels. The method is designed to be transferable once wheel-specific images are available.

### 2. Exploratory Data Analysis
- Check class distribution and imbalance
- Visualize sample OK vs. defective images
- Verify image size, color channels, and data quality

### 3. Modeling
- **Baseline:** a simple custom CNN as a reference point
- **Transfer learning:** fine-tune ImageNet-pretrained models and compare them: MobileNetV2 [[8]](#-references), EfficientNet-B0 [[9]](#-references), and ResNet50 [[10]](#-references). Transfer learning reduces the amount of training data needed [[11]](#-references) and has been applied successfully to casting defect detection [[1]](#-references), including on this same impeller dataset [[6]](#-references) [[7]](#-references)
- **Data augmentation:** rotation, flips, and brightness changes to simulate real line conditions

### 4. Evaluation from an Industrial Perspective
Accuracy alone is misleading for quality inspection. This project focuses on:
- **Recall on the defective class** as the main metric: a missed defect (*false negative*) is far more costly than re-checking a good part (*false positive*)
- Confusion matrix, precision, recall, and F1-score
- **Threshold tuning** to balance missed defects vs. false alarms, with the trade-off explained
- Inference time per image, to check whether it can keep up with production speed

### 5. Explainability (Grad-CAM)
Use **Grad-CAM** [[12]](#-references) to generate heatmaps that highlight the region the model considers defective, so QC operators can **verify and trust** the model's decision instead of treating it as a black box.

### 6. Demo Application
A **Streamlit** web app where a user can:
- Upload a product image
- Get the result **OK / Defective** with a confidence score
- See the Grad-CAM heatmap
- *(Optional)* View a simple inspection log and daily summary (total inspected, defect rate)

---

## 🏭 Path to Real Production (Wheel Inspection)

This project is a proof of concept. Applying it to an actual alloy wheel production line would require:

1. **Collecting real wheel images** from the line under consistent lighting and camera positioning
2. **Labeling specific defect types** (porosity, shrinkage, cracks, surface defects) with QC experts
3. Moving from classification to **object detection / segmentation** (e.g. YOLO or Mask R-CNN) to locate each defect, as demonstrated for aluminum casting inspection in [[1]](#-references) and [[4]](#-references)
4. **Edge deployment** (ONNX / TFLite on an industrial PC or edge device) next to the conveyor
5. **Integration with production systems (MES)** so every inspection result is recorded for traceability and defect trend analysis

---

## 🧰 Planned Tech Stack

`Python` · `TensorFlow / Keras` *(or PyTorch)* · `OpenCV` · `scikit-learn` · `Grad-CAM` · `Streamlit` · `Google Colab`

## 📁 Planned Repository Structure

```
casting-defect-detection/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation_gradcam.ipynb
├── app/
│   └── streamlit_app.py
├── models/            # trained model files
├── reports/figures/   # confusion matrix, training curves, Grad-CAM samples
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- [x] Exploratory data analysis
- [ ] Baseline CNN
- [ ] Transfer learning and model comparison
- [ ] Evaluation (recall-focused) and threshold tuning
- [ ] Grad-CAM explainability
- [ ] Streamlit demo app
- [ ] Deploy demo online
- [ ] Final results and documentation

## 📊 Results

### EDA Summary ([notebook](notebooks/01_eda.ipynb))

The dataset contains **7,348 images**, split into **6,633 for training** and **715 for testing**. Defective products slightly outnumber OK products, so the dataset is mildly imbalanced. To prevent the model from favoring the majority class, training will use **class weights**, and evaluation will focus on **recall for the defective class** rather than accuracy alone.

Visually, **OK products** have clean, well-defined outer and inner rims with few scratches or small surface marks. **Defective products** show damage on the outer rim, small chips and scratches across the surface, and **irregular, non-circular shapes**. Most differences between the two classes appear on the **outer and inner rims**, which makes these areas the clearest visual indicators of a defect. Grad-CAM will be used later to check whether the model also focuses on these regions.

| | |
|---|---|
| ![Class distribution](reports/figures/01_class_distribution.png) | ![Average images](reports/figures/04_average_images.png) |
| ![OK samples](reports/figures/02_samples_ok.png) | ![Defective samples](reports/figures/02_samples_defective.png) |

### Model Results

*To be added after training and evaluation.*

---

## 📚 References

**Casting defect detection & industrial inspection**

1. M. Ferguson, R. Ak, Y.-T. T. Lee, and K. H. Law, "Detection and Segmentation of Manufacturing Defects with Convolutional Neural Networks and Transfer Learning," *Smart and Sustainable Manufacturing Systems*, vol. 2, no. 1, 2018. [doi:10.1520/SSMS20180033](https://doi.org/10.1520/SSMS20180033) · [arXiv:1808.02518](https://arxiv.org/abs/1808.02518)
2. T. Czimmermann et al., "Visual-Based Defect Detection and Classification Approaches for Industrial Applications—A SURVEY," *Sensors*, vol. 20, no. 5, p. 1459, 2020. [doi:10.3390/s20051459](https://doi.org/10.3390/s20051459)
3. D. Mery, "Aluminum Casting Inspection Using Deep Learning: A Method Based on Convolutional Neural Networks," *Journal of Nondestructive Evaluation*, vol. 39, 2020. [doi:10.1007/s10921-020-0655-9](https://doi.org/10.1007/s10921-020-0655-9)
4. D. Mery, "Aluminum Casting Inspection using Deep Object Detection Methods and Simulated Ellipsoidal Defects," *Machine Vision and Applications*, vol. 32, art. 72, 2021.

**Dataset & prior work on the same dataset**

5. R. Dabhi, "Casting product image data for quality inspection" (Real-life Industrial Dataset of Casting Product), Kaggle, 2020. [Link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
6. M. Habibpour et al., "An Uncertainty-Aware Deep Learning Framework for Defect Detection in Casting Products," 2021. [arXiv:2107.11643](https://arxiv.org/abs/2107.11643)
7. N. Perišić and R. Jovanović, "Application of deep learning in quality inspection of casting products," University of Belgrade, Faculty of Mechanical Engineering, 2022. [Link](https://machinery.mas.bg.ac.rs/handle/123456789/4498)

**Model architectures & methods**

8. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
9. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
10. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
11. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," *NeurIPS*, 2014. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
12. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," *ICCV*, 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

---

## 🙏 Acknowledgements

Dataset: *Real-life Industrial Dataset of Casting Product* by Ravirajsinh Dabhi, published on Kaggle. Please refer to the dataset page for its license and terms of use.

## 👤 Author

**Gibral Haikal Faiz**, Industrial Informatics Engineering Student, Politeknik Manufaktur Bandung
[LinkedIn](https://www.linkedin.com/in/gibral-haikal-faiz-46449a319/) · [GitHub](https://github.com/gbrlhf)
