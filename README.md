# Casting Defect Detection

**AI-based visual quality inspection for metal casting products — classifying parts as OK or Defective, with explainable results for QC operators.**

> **Status: In progress.** This README describes the problem and the planned solution. Results will be added once the model is trained and evaluated.

---

## Background

Metal casting is a core process in automotive component manufacturing, from engine parts to **aluminum alloy wheels**, which are commonly produced by *low pressure die casting*. During casting, molten metal can solidify imperfectly and leave defects such as:

- **Porosity / blow holes**: gas trapped inside the metal
- **Shrinkage**: cavities caused by uneven cooling
- **Cracks and pinholes**
- **Surface irregularities**: burrs, flash, or rough surfaces

A defective part that reaches the customer can lead to **product failure, warranty claims, and rejected shipments**. For safety-critical parts like wheels, this risk is unacceptable.

## Problem Statement

In many factories, final inspection is still done **manually by human operators**, which has several limitations:

| Problem | Impact |
|---|---|
| **Slow** compared to production speed | Inspection becomes a bottleneck on high-volume lines |
| **Subjective**: results vary between operators and shifts | Inconsistent quality standards |
| **Fatigue**: accuracy drops over long shifts | Defective parts slip through (false negatives) |
| **No data trail**: decisions are not recorded systematically | Hard to analyze defect trends and root causes |

**Goal:** build a model that can automatically and consistently classify casting products as **OK** or **Defective** from images, and explain *why* it made each decision, so that it can support (not replace) human inspectors.

---

## Proposed Solution

```
 Product image ──► Preprocessing ──► CNN Classifier ──► OK / Defective + confidence
                                          │
                                          └──► Grad-CAM heatmap (where the defect is)
                                                     │
                                                     ▼
                                    Web demo for QC operator + inspection log
```

### 1. Dataset
Using the public **[Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)** (Kaggle), which contains top-view grayscale images of cast **submersible pump impellers**, labeled as:
- `ok_front`: good product
- `def_front`: defective product

> **Note:** the images are pump impellers, not wheels. They are used here because they show **real casting defects** (porosity, blow holes, shrinkage) of the same kind found in cast aluminum wheels. The method is designed to be transferable once wheel-specific images are available.

### 2. Exploratory Data Analysis
- Check class distribution and imbalance
- Visualize sample OK vs. defective images
- Verify image size, color channels, and data quality

### 3. Modeling
- **Baseline:** a simple custom CNN as a reference point
- **Transfer learning:** fine-tune pretrained models (e.g. MobileNetV2, EfficientNet-B0, ResNet50) and compare them
- **Data augmentation:** rotation, flips, and brightness changes to simulate real line conditions

### 4. Evaluation from an Industrial Perspective
Accuracy alone is misleading for quality inspection. This project focuses on:
- **Recall on the defective class** as the main metric: a missed defect (*false negative*) is far more costly than re-checking a good part (*false positive*)
- Confusion matrix, precision, recall, and F1-score
- **Threshold tuning** to balance missed defects vs. false alarms, with the trade-off explained
- Inference time per image, to check whether it can keep up with production speed

### 5. Explainability (Grad-CAM)
Generate heatmaps that highlight the region the model considers defective, so QC operators can **verify and trust** the model's decision instead of treating it as a black box.

### 6. Demo Application
A **Streamlit** web app where a user can:
- Upload a product image
- Get the result **OK / Defective** with a confidence score
- See the Grad-CAM heatmap
- *(Optional)* View a simple inspection log and daily summary (total inspected, defect rate)

---

## Path to Real Production (Wheel Inspection)

This project is a proof of concept. Applying it to an actual alloy wheel production line would require:

1. **Collecting real wheel images** from the line under consistent lighting and camera positioning
2. **Labeling specific defect types** (porosity, shrinkage, cracks, surface defects) with QC experts
3. Moving from classification to **object detection / segmentation** (e.g. YOLO) to locate each defect
4. **Edge deployment** (ONNX / TFLite on an industrial PC or edge device) next to the conveyor
5. **Integration with production systems (MES)** so every inspection result is recorded for traceability and defect trend analysis

---

## Planned Tech Stack

`Python` · `TensorFlow / Keras` *(or PyTorch)* · `OpenCV` · `scikit-learn` · `Grad-CAM` · `Streamlit` · `Google Colab`

## Planned Repository Structure

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

## Results

### EDA Summary ([notebook](notebooks/01_eda.ipynb))

The dataset contains **7,348 images**, split into **6,633 for training** and **715 for testing**. Defective products slightly outnumber OK products, so the dataset is mildly imbalanced. To prevent the model from favoring the majority class, training will use **class weights**, and evaluation will focus on **recall for the defective class** rather than accuracy alone.

Visually, **OK products** have clean, well-defined outer and inner rims with few scratches or small surface marks. **Defective products** show damage on the outer rim, small chips and scratches across the surface, and **irregular, non-circular shapes**. Most differences between the two classes appear on the **outer and inner rims**, which makes these areas the clearest visual indicators of a defect. Grad-CAM will be used later to check whether the model also focuses on these regions.

| | |
|---|---|
| ![Class distribution](reports/figures/01_class_distribution.png) | ![Average images](reports/figures/04_average_images.png) |
| ![OK samples](reports/figures/02_samples_ok.png) | ![Defective samples](reports/figures/02_samples_defective.png) |

### TRAINING RESULT ([notebook](notebooks/02_training.ipynb))

Four models were compared on the 715-image test set. All three transfer-learning models reached around 99% accuracy, far above the baseline CNN trained from scratch (~84% accuracy), showing that ImageNet-pretrained features transfer well to casting inspection.

ResNet50V2 was selected as the best model because it achieved the highest defective recall (fewest missed defects) together with the best F1-score. MobileNetV2 and EfficientNet-B0 reached slightly higher precision but missed a few more defective parts. Since a missed defect is costlier than a false alarm, recall was prioritized.

Trade-off: ResNet50V2 is also the largest and slowest model. For deployment on an edge device next to a production line, MobileNetV2 remains a strong alternative with nearly identical performance at a fraction of the size.

### Model Comparison ([notebook](notebooks/02_training.ipynb))

Four models were evaluated on the **715-image test set** (453 defective, 262 OK) at the default threshold of 0.5:

| Model | Accuracy | Precision | Recall (defective) | F1 | AUC | Missed defects | False alarms | ms / image | Params (M) |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet50V2** ✅ | **99.58%** | 99.56% | **99.78%** | **99.67%** | 0.9999 | **1** | 2 | 5.19 | 23.57 |
| MobileNetV2 | **99.58%** | **100%** | 99.34% | **99.67%** | 1.0000 | 3 | **0** | 2.87 | 2.26 |
| EfficientNet-B0 | 99.02% | **100%** | 98.45% | 99.22% | 0.9986 | 7 | **0** | 4.78 | 4.05 |
| Baseline CNN | 83.92% | 83.01% | 93.82% | 88.08% | 0.9105 | 28 | 87 | **0.57** | **0.09** |

*Missed defects and false alarms are derived from recall, precision, and the test set size. Inference time measured on a Google Colab T4 GPU.*

![Model comparison](reports/figures/07_model_comparison.png)

**Key findings**

- **Transfer learning makes a large difference.** All three pretrained models reached about 99% accuracy, compared with 83.9% for the baseline CNN trained from scratch. The baseline let 28 defective parts through and raised 87 false alarms.
- **ResNet50V2 was selected as the best model.** It tied with MobileNetV2 on accuracy and F1, but had the **highest defective recall**, missing only **1 of 453** defective parts. Since a missed defect is costlier than a false alarm, recall decided the tie.
- **MobileNetV2 is the practical alternative for deployment.** It missed 2 more defects than ResNet50V2 but produced **zero false alarms**, while being about **10× smaller** and **1.8× faster**. For an edge device next to a production line, this trade-off may be worth it; the final choice depends on the factory's cost of a missed defect versus a false alarm.

> ⚠️ **Note:** near-perfect scores are common on this dataset, which is clean and was augmented by its author, so similar images can appear in both train and test. Performance on a real production line with different lighting and parts would need to be validated with new data.

---

## References

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

## Acknowledgements

Dataset: *Real-life Industrial Dataset of Casting Product* by Ravirajsinh Dabhi, published on Kaggle. Please refer to the dataset page for its license and terms of use.

## Author

**Gibral Haikal Faiz**, Industrial Informatics Engineering Student, Politeknik Manufaktur Bandung
[LinkedIn](https://www.linkedin.com/in/gibral-haikal-faiz-46449a319/) · [GitHub](https://github.com/gbrlhf)