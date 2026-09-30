# CARLA Semantic Segmentation: U-Net vs SegFormer-B0

A semantic segmentation benchmark for autonomous driving scenes generated with the CARLA simulator.

This project compares a custom lightweight **U-Net** architecture with **SegFormer-B0** for three-class semantic segmentation:

- Background
- Road
- Lane

The models are evaluated on a common evaluation subset using segmentation quality metrics and pure model inference latency. A Gradio-based Hugging Face interface is also provided for interactive inference on individual images.

---

## Project Overview

Semantic segmentation is an important computer vision task for autonomous driving systems because it provides pixel-level information about the driving environment.

In this project, CARLA-generated RGB images are segmented into three classes:

| Class ID | Class |
|---|---|
| 0 | Background |
| 1 | Road |
| 2 | Lane |

Two different segmentation approaches are compared:

1. **Custom U-Net**
   - CNN-based encoder-decoder architecture
   - Designed specifically for the three-class CARLA segmentation task
   - 32 → 64 → 128 → 256 feature channels

2. **SegFormer-B0**
   - Transformer-based semantic segmentation architecture
   - Initialized from `nvidia/segformer-b0-finetuned-ade-512-512`
   - The segmentation head was adapted from 150 ADE20K classes to 3 CARLA classes

The primary objective is to examine the trade-off between **segmentation quality and inference efficiency**, with particular attention to the thin lane class.

---

## Dataset

The dataset was generated using the CARLA autonomous driving simulator.

### Dataset Size

- Total images: **38,053**
- Total masks: **38,053**
- Resolution: **640 × 384**
- Number of classes: **3**

### Classes

```text
0 → Background
1 → Road
2 → Lane
````

The lane class is considerably thinner and less dominant than the background and road classes. Therefore, class imbalance is an important characteristic of this dataset.

---

## Data Preprocessing

### U-Net

RGB images were converted to floating-point tensors and scaled from:

```text
[0, 255] → [0, 1]
```

No ImageNet normalization was used for U-Net.

### SegFormer

SegFormer uses the Hugging Face `AutoImageProcessor` associated with the pretrained ADE20K checkpoint.

The model was fine-tuned for the three CARLA classes while maintaining the preprocessing pipeline associated with the pretrained model.

During training and evaluation, SegFormer logits were resized to the target resolution of:

```text
640 × 384
```

before calculating the loss and segmentation metrics.

---

# U-Net

## Architecture

The U-Net implementation uses a compact convolutional encoder-decoder architecture.

The encoder consists of:

```text
Input
  ↓
DoubleConv(3 → 32)
  ↓
MaxPool
  ↓
DoubleConv(32 → 64)
  ↓
MaxPool
  ↓
DoubleConv(64 → 128)
  ↓
MaxPool
  ↓
DoubleConv(128 → 256)
```

The decoder uses transposed convolutions and skip connections:

```text
256 → 128
      ↓
     Skip connection
      ↓
128 → 64
      ↓
     Skip connection
      ↓
64 → 32
      ↓
     Skip connection
      ↓
3-class output
```

Each `DoubleConv` block contains:

* 3×3 convolution
* Batch Normalization
* ReLU
* 3×3 convolution
* Batch Normalization
* ReLU

---

## U-Net Training

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| Epochs            | 15                     |
| Batch size        | 4                      |
| Optimizer         | Adam                   |
| Learning rate     | 1e-3                   |
| Loss              | Weighted Cross Entropy |
| Input             | RGB                    |
| Number of classes | 3                      |

Class weights:

```text
Background: 1
Road:       5
Lane:       50
```

The high weight assigned to the lane class was used to compensate for the strong class imbalance and encourage the model to preserve thin lane structures.

### Checkpoint

```text
models/unet_3class_v1.pth
```

---

# SegFormer-B0

The second model is based on:

```text
nvidia/segformer-b0-finetuned-ade-512-512
```

The original model is trained for semantic segmentation on ADE20K with 150 classes.

For this project, the segmentation head was adapted to:

```text
3 classes
```

using:

```python
num_labels=3
ignore_mismatched_sizes=True
```

Therefore, the pretrained encoder/decoder representation is reused while the original 150-class segmentation head is replaced for the CARLA task.

---

## SegFormer Training

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| Epochs            | 15                     |
| Batch size        | 4                      |
| Optimizer         | AdamW                  |
| Learning rate     | 6e-5                   |
| Weight decay      | 0.01                   |
| Scheduler         | Cosine                 |
| Warmup            | 10%                    |
| Loss              | Weighted Cross Entropy |
| Number of classes | 3                      |

Class weights:

```text
Background: 1
Road:       5
Lane:       50
```

### Evaluation Split

A common evaluation subset was first removed from the full dataset.

The remaining samples were then divided into:

```text
90% training
10% validation
```

The split was reproducible using a fixed random seed.

The best SegFormer checkpoint was selected according to validation mIoU.

### Best Checkpoint

```text
Epoch: 13
Validation mIoU: 0.89238
```

Checkpoint:

```text
models/segformer_b0_carla_best.pth
```

---

# Evaluation Methodology

## Common Evaluation Benchmark

Because the original U-Net training split was not saved, its exact original train/validation partition cannot be reconstructed.

Therefore, a new reproducible common evaluation subset was created and used to compare both models.

The benchmark contains:

```text
7,610 images
```

which corresponds to approximately 20% of the complete dataset.

The indices were generated using:

```text
Random seed: 42
```

and stored in:

```text
evaluation/eval_indices.npy
```

The same images are evaluated by both models.

### Important Evaluation Limitation

This subset should **not** be interpreted as a strict held-out test set for both models.

The original U-Net split was not preserved, so some of the 7,610 benchmark images may have been included in the original U-Net training set.

The SegFormer training procedure explicitly excluded this common evaluation subset.

Therefore, the benchmark is best described as a:

> **Common-input comparative benchmark**

rather than a clean independent test set.

This distinction is important when interpreting the reported performance.

---

# Evaluation Metrics

The following segmentation metrics are reported.

### Pixel Accuracy

The proportion of correctly classified pixels over all pixels.

### Intersection over Union (IoU)

For a class:

```text
IoU = TP / (TP + FP + FN)
```

The mean IoU (mIoU) is the mean IoU across all three classes.

### Dice Score

For a class:

```text
Dice = 2TP / (2TP + FP + FN)
```

The reported Mean Dice is the mean Dice score across all three classes.

Because the lane class occupies a relatively small portion of the image, **lane IoU is particularly informative** for this comparison.

---

# Benchmark Results

Both models were evaluated on the same 7,610-image common evaluation subset.

## Overall Results

| Metric                 |           U-Net |       SegFormer-B0 |
| ---------------------- | --------------: | -----------------: |
| Pixel Accuracy         |      **99.69%** |             98.73% |
| mIoU                   |      **96.57%** |             87.67% |
| Mean Dice              |      **98.21%** |             92.74% |
| Lane IoU               |      **90.72%** |             67.06% |
| Lane Dice              |      **95.13%** |             80.28% |
| Pure inference latency | 10.209 ms/image | **5.525 ms/image** |
| Corresponding FPS      |           97.96 |         **181.01** |

---

## Per-Class Results

### U-Net

| Class      |    IoU |   Dice |
| ---------- | -----: | -----: |
| Background | 99.78% | 99.89% |
| Road       | 99.19% | 99.60% |
| Lane       | 90.72% | 95.13% |

### SegFormer-B0

| Class      |    IoU |   Dice |
| ---------- | -----: | -----: |
| Background | 99.26% | 99.63% |
| Road       | 96.68% | 98.31% |
| Lane       | 67.06% | 80.28% |

---

# Performance Analysis

The benchmark demonstrates a clear quality-efficiency trade-off between the two models.

### Segmentation Quality

U-Net achieves higher segmentation quality across the reported metrics.

The largest difference occurs in the lane class:

```text
U-Net lane IoU:       90.72%
SegFormer lane IoU:   67.06%

Difference:           23.66 percentage points
```

The overall mIoU difference is:

```text
U-Net:                96.57%
SegFormer-B0:         87.67%

Difference:            8.90 percentage points
```

This indicates that the custom U-Net preserves the thin lane structures substantially better on this CARLA benchmark.

### Inference Efficiency

SegFormer-B0 has considerably lower pure model inference latency:

```text
U-Net:         10.209 ms/image
SegFormer-B0:   5.525 ms/image
```

The corresponding throughput values are:

```text
U-Net:          97.96 FPS
SegFormer-B0:  181.01 FPS
```

Thus, SegFormer-B0 provides substantially higher inference throughput under the benchmark configuration.

### Overall Interpretation

The results can therefore be summarized as a quality-efficiency trade-off:

```text
U-Net
↑ segmentation quality
↑ lane segmentation quality
↓ inference speed

SegFormer-B0
↓ segmentation quality on this dataset
↓ lane segmentation quality
↑ inference speed
```

These results are specific to the trained models, dataset, preprocessing pipeline, hardware, and benchmark configuration. They should not be generalized to U-Net or SegFormer architectures in general.

---

# Confusion Matrices

## U-Net

```text
[[1145789444    2137374     140009]
 [    214617  685883332    3054407]
 [      1648     164054   32848715]]
```

## SegFormer-B0

```text
[[1141457697    6090157     518973]
 [   1828975  673058600   14264781]
 [     31958     927865   32054594]]
```

The largest difference is associated with the lane class.

For ground-truth lane pixels:

```text
U-Net lane → road:        164,054
SegFormer lane → road:    927,865
```

This is consistent with the substantially lower lane IoU obtained by SegFormer-B0.

---

# Inference Latency

Latency was measured separately from image preprocessing and postprocessing.

The timed section corresponds to the model forward pass.

For CUDA execution, synchronization was performed before and after the forward pass to avoid asynchronous GPU execution from producing misleading measurements.

Conceptually:

```python
if DEVICE.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

outputs = model(x)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

end = time.perf_counter()
```

The following operations are excluded from the reported pure model latency:

* image loading
* image preprocessing
* CPU → GPU transfer
* mask visualization
* overlay generation
* disk I/O

Therefore, the reported value should be interpreted as **pure model forward inference latency**.

---

## Batch Size

The benchmark used:

```text
Batch size = 4
```

The reported latency is therefore an amortized per-image forward latency under the batch-4 configuration.

The Hugging Face interactive application uses:

```text
Batch size = 1
```

because the application processes one uploaded image at a time.

Consequently, latency measured in the interactive UI may differ from the benchmark values.

The UI measurement is more representative of single-image interactive inference, while the benchmark measures throughput under the specified batch configuration.

---

# Hugging Face Demo

The project includes a Gradio-based interface designed for interactive inference.

The user can:

1. Upload a CARLA image. (15 random images are included in the repository for testing purposes.)
2. Select a model.
3. Run inference.
4. View the predicted segmentation mask.
5. View an overlay of the prediction on the original image.
6. View pure model inference latency.
7. View the corresponding FPS.

Supported models:

```text
U-Net
SegFormer-B0
```

The interface is intended to provide a simple visual comparison between the two models without requiring the user to run the benchmark manually.

---

# Repository Structure

```text
carla-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── unet_3class_v1.pth
│   └── segformer_b0_carla_best.pth
│
├── evaluation/
│   └── eval_indices.npy
│
└── test_images/
    ├── image_01.png
    ├── image_02.png
    ├── image_03.png
    ├── ...
    └── image_15.png

```

The exact repository structure may vary depending on how the training and evaluation scripts are organized.

---

# Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
torchvision
transformers
gradio
pillow
opencv-python-headless
numpy
```

---

# Running the Gradio Application

From the project root:

```bash
python app.py
```

The application will start a local Gradio interface.

Upload an image, select the desired model, and click:

```text
Run Inference
```

The interface returns the predicted mask, segmentation overlay, latency, and corresponding FPS.

---

# Hardware
(RTX 3070 8GB)
Inference performance is hardware-dependent.

The reported latency values should therefore be interpreted together with the hardware and software environment used during benchmarking.

Changing any of the following can significantly affect latency:

* GPU model
* CPU model
* CUDA version
* PyTorch version
* batch size
* input resolution
* precision
* driver version
* memory state
* warm-up configuration

Therefore, the reported FPS values are benchmark measurements rather than architecture-independent performance claims.

---

# Limitations

Several limitations should be considered when interpreting the results.

### 1. Original U-Net split is unavailable

The original U-Net training split was created using an 80/20 random split, but the original random seed and exact indices were not preserved.

As a result, the exact original U-Net validation/test partition cannot be reconstructed.

### 2. Common benchmark is not a clean test set

The 7,610-image common benchmark may contain images that were previously seen by U-Net during its original training.

Therefore, the benchmark should not be described as a completely unseen test set for both models.

### 3. Dataset Domain

The images are generated by CARLA.

Performance on simulated images does not directly establish equivalent performance on real-world autonomous-driving datasets.

### 4. Three-Class Segmentation

The task is intentionally simplified to:

```text
Background / Road / Lane
```

Real autonomous-driving perception systems generally require substantially more semantic classes and often combine multiple sensors and perception tasks.

### 5. Latency Depends on Hardware

The reported latency values are environment-specific.

A different GPU, CPU, batch size, or inference configuration can produce different results.

---

# Future Work

Potential extensions include:

* evaluation on a strictly held-out dataset
* repeated experiments with multiple random seeds
* evaluation on additional CARLA weather and lighting conditions
* cross-domain evaluation on real-world road datasets
* mixed-precision inference
* TensorRT optimization
* ONNX export
* edge-device deployment
* Raspberry Pi / embedded GPU benchmarking
* INT8 quantization
* model pruning
* knowledge distillation
* comparison with additional lightweight segmentation models
* temporal consistency between consecutive frames
* integration with autonomous-driving planning and control
* comparison of segmentation quality versus end-to-end driving performance

A particularly relevant direction is evaluating whether the additional segmentation quality provided by U-Net translates into measurable improvements in downstream lane-following or trajectory-control performance.

---

# Conclusion

This project compares a task-specific CNN-based U-Net with a pretrained SegFormer-B0 transformer model for three-class semantic segmentation of CARLA driving scenes.

On the common evaluation benchmark:

* **U-Net achieved higher segmentation quality**, especially for the lane class.
* **SegFormer-B0 achieved substantially lower pure inference latency and higher throughput**.
* The largest quality difference occurred in lane segmentation, where U-Net achieved 90.72% IoU compared with 67.06% for SegFormer-B0.
* SegFormer-B0 reduced measured pure inference latency from 10.209 ms/image to 5.525 ms/image under the benchmark configuration.

The results therefore illustrate a practical trade-off between **pixel-level segmentation quality and inference efficiency** for autonomous-driving perception workloads.

The comparison is specific to this CARLA dataset, training procedure, checkpoints, preprocessing pipeline, and benchmark environment.

---

## License

Add the license appropriate for your source code and any third-party components before publishing the repository.

Note that the CARLA simulator, pretrained model weights, and datasets may have their own licensing and usage conditions. Check the applicable licenses before redistributing data or model weights.