# Steel Surface Defect Classification with ResNet18

Week 6 lab of *Introduction to Applied AI* (BS 8th Semester, Department of Electronics): a complete computer vision pipeline that classifies six types of surface defect on strip steel, with a data audit, a leakage check, macro-F1 model selection and a Grad-CAM shortcut check.

**Result in one line:** a fine-tuned ResNet18 reaches **macro-F1 0.997** on 360 held-out test images, with one error out of 360.

## Contents

- [Why this matters](#why-this-matters)
- [Dataset](#dataset)
- [Data audit](#data-audit)
- [Method](#method)
- [Results](#results)
- [Grad-CAM shortcut check](#grad-cam-shortcut-check)
- [Limitations](#limitations)
- [Responsibility questions](#responsibility-questions)
- [How to reproduce](#how-to-reproduce)
- [Repository layout](#repository-layout)

## Why this matters

**Who benefits:** quality-control engineers and inspectors on steel production lines, who must find surface defects in strip steel.

**Why it matters:** defects that go unnoticed lead to scrap, rework, and defective product reaching customers. Checking images by eye is slow and differs from one inspector to the next, so an automatic first-pass classifier can help sort and flag images for a person to review.

## Dataset

| | |
|---|---|
| **Name** | NEU Surface Defect Database (the `NEU-DET` folder), Kaggle upload by `kaustubhdikshit` |
| **Link** | https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database |
| **Size** | 1,800 images, 200 x 200 pixels, stored as RGB but visually grayscale |
| **Classes (6)** | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| **Labels** | Class folders supplied by the dataset's creators; I did not re-label or verify them independently |
| **Licence** | See the dataset's Kaggle page for its terms. The images are **not** redistributed in this repository |

How the data was used:

| Split | Images | Source |
|---|---|---|
| Train | 1,200 | 5/6 of the `train/images` folder (1,440 images), split by group |
| Validation | 240 | 1/6 of the `train/images` folder, used to choose the best epoch |
| Test | 360 | the dataset's `validation/images` folder (60 per class), used once at the end |

## Data audit

| Check | Finding |
|---|---|
| Class counts (train folder) | 240 images in every class |
| Imbalance ratio (max / min) | 1.0, perfectly balanced |
| Image size and mode | 200 x 200 pixels, RGB (300 of 300 sampled) |
| Images per group | 1.00: file names carry no part or subject ID, so every image is its own group |
| Shared groups, train vs validation | 0 |
| Exact duplicate files (MD5, train and test combined) | 2 files, i.e. one identical pair. The notebook did not remove them, so at most one test image could be affected |
| Test images with a near-duplicate in train | 0 of 360 (threshold 0.10; closest distances 0.108, 0.115, 0.115, 0.126, 0.126) |
| Flagged pairs inspected | I viewed the 4 closest train/test pairs side by side: they are different images with similar steel texture and lighting, so nothing was deleted |

**First shortcut suspect: overall brightness.** In the sample grid, `patches` and `pitted_surface` images are bright, almost white, while `inclusion` and `scratches` images are dark grey. Brightness is not the defect itself, so a model could partly rely on it. This is only a suspect: `crazing` images range from dark to bright, and several images have a lighting gradient from left to right.

![Sample images from every class](images/sample_grid.png)

## Method

- **Backbone:** ResNet18 pretrained on ImageNet, with the last layer replaced by a 6-class head. Images are converted to grayscale and then to 3 channels, and resized to 224 x 224.
- **Imbalance handling:** weighted cross-entropy with weight = N / (K x n_c). Training set N = 1,200 and K = 6 classes, so for `inclusion` (n_c = 199) the weight is 1200 / (6 x 199) = 1.005, and for `patches` (n_c = 201) it is 0.995. All weights are between 0.995 and 1.005, because the classes are balanced, so imbalance is not a problem in this dataset. I did not try the balanced sampler.
- **Augmentation:** random resized crop (scale 0.8 to 1.0), rotation up to 10 degrees, and brightness/contrast jitter of 0.15. Flips and strong distortions were left out to keep defects realistic. The jitter may also reduce reliance on brightness.
- **Training:** two phases, about 5 seconds per epoch on a Kaggle GPU. Phase 1 trained only the head (3,078 parameters) for 2 epochs. Phase 2 also fine-tuned the last block (`layer4`) for 3 epochs. The best epoch was chosen by validation **macro-F1**, not accuracy.
- **Split:** `StratifiedGroupKFold` (6 splits, seed 42), so no group appears in both train and validation.

## Results

### Validation macro-F1 during training

| Phase | Epoch | Validation macro-F1 |
|---|---|---|
| 1 (head only) | 1 | 0.744 |
| 1 (head only) | 2 | 0.975 |
| 2 (fine-tune last block) | 1 | 0.987 |
| 2 (fine-tune last block) | 2 | 0.996 |
| 2 (fine-tune last block) | 3 | 1.000 |

Fine-tuning helped, raising the best validation macro-F1 from 0.975 to 1.000. A perfect score on 240 validation images suggests the task is fairly easy, and the best epoch was chosen on that same set, so it is slightly optimistic.

### Test set (used once): 360 images

**Macro-F1 0.997, accuracy 0.997.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| crazing | 1.000 | 1.000 | 1.000 | 60 |
| inclusion | 1.000 | 0.983 | 0.992 | 60 |
| patches | 1.000 | 1.000 | 1.000 | 60 |
| pitted_surface | 1.000 | 1.000 | 1.000 | 60 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 | 60 |
| scratches | 0.984 | 1.000 | 0.992 | 60 |

The only error is one `inclusion` image predicted as `scratches`. Both defects can appear as elongated streaks, so the confusion is plausible. One error is 0.3% of the test set, so the score should not be read as a guarantee on a real production line.

Binary metrics (a recall threshold, sensitivity, specificity and PPV) do not apply, because this task has six classes.

![Confusion matrix on the test set](images/confusion_matrix.png)

## Grad-CAM shortcut check

Grad-CAM highlights the image regions that drove each prediction. Because the model made only one mistake, the grid shows four heatmaps (three correct, one wrong) and two empty panels.

![Grad-CAM heatmaps](images/gradcam.png)

1. True crazing, predicted crazing: two hot spots along the top, on crack junctions in the surface texture. Legitimate evidence.
2. True crazing, predicted crazing: a vertical band on the left and a horizontal band across the top, following the crack network. Legitimate.
3. True crazing, predicted crazing: one hot spot at the upper left, on a crack junction. Legitimate, though less spread out.
4. True inclusion, predicted scratches (the one error): attention on a dark vertical streak at the left. The model looked at the real defect, but its long, thin shape resembles a scratch.
5. and 6. Not shown, because there were no more wrong examples.

**Verdict: no shortcut found, with limited confidence.** The attention sits on the defect and the surface texture, not on borders, corners or the brightness of the whole image, and the images contain no text or labels. The sample is small, and all three correct examples are the same class.

**Follow-up experiment:** for one correctly classified image per class, mask or crop out the highlighted region and re-score. A large drop in confidence would show the model depends on that region. Brightness could also be tested directly by darkening or brightening images and checking whether predictions change.

## Limitations

- **Data source:** one dataset from a controlled setting: 200 x 200 images of steel surfaces under consistent lighting, six defect types, one defect type per image, and no defect-free class.
- **Evaluation:** a small test set (360 images), a best epoch selected on a validation set drawn from the same data, one training run, and only four Grad-CAM heatmaps inspected.
- **Not for:** automatic accept/reject decisions on steel, safety-critical parts, or any production line other than the one that produced these images, without further validation and human review.

## Responsibility questions

**1. Who could be harmed and how?** Customers and downstream manufacturers could receive defective steel if the model misses or mislabels a defect. A wrong label also sends the wrong signal to the process team: an inclusion called scratches points to a different cause than an inclusion called inclusion, so effort could go to the wrong fix. Because the model has no "no defect" class, a clean strip would be forced into one of the six defect labels, causing false alarms and wasted rework.

**2. Where was the model not validated?** Only on this one dataset: one camera and lighting setup, 200 x 200 images and six defect types. It has not been tested on other steel grades, mills, cameras, lighting, image sizes, strips with no defect, or strips with several defects.

**3. What is the human role?** An inspector stays responsible for the accept/reject decision. The model works as an assistant that sorts and flags images, and a person reviews its predictions, especially uncertain ones and the defect types that get confused (inclusion and scratches). People should also audit a sample of predictions regularly and report new defect types or changes on the line.

**4. What evidence would change my conclusions?** Lower scores on images from a different line, camera or lighting would show the result does not transfer. A drop in accuracy after darkening or brightening the images, or after masking the Grad-CAM region, would show the model relies on brightness or a narrow area. Heatmaps that concentrate on borders or corners would confirm a shortcut. More errors on a larger test set, or duplicates between train and test found by a deeper audit, would mean the 0.997 macro-F1 is too optimistic.

## How to reproduce

1. On [Kaggle](https://www.kaggle.com), create a notebook, click **Add Input**, and add the *NEU Surface Defect Database* dataset linked above.
2. In the notebook settings, turn on a **GPU** accelerator and **Internet** (needed to download the pretrained ResNet18 weights).
3. Upload the notebook from this repository (**File → Import Notebook**) or copy its cells in order.
4. Check the configuration cell. The settings used here were:

```python
DATA_ROOT = '/kaggle/input/datasets/kaustubhdikshit/neu-surface-defect-database/NEU-DET'
TRAIN_DIR = f'{DATA_ROOT}/train/images'
TEST_DIR  = f'{DATA_ROOT}/validation/images'   # held-out test set, used once
GRAYSCALE = True
IMG_SIZE  = 224
BACKBONE  = 'resnet18'
GROUP_PATTERN = None    # file names contain no subject ID
POSITIVE = None         # six classes, so no binary threshold
TARGET_RECALL = None
```

   The dataset path can differ between Kaggle sessions, so confirm it in the **Input** panel.
5. Click **Run All**. Training takes roughly a minute. The final cell saves `week6_model.pt` (ResNet18 weights, class list, image size) to `/kaggle/working`.

Libraries: PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, matplotlib (all preinstalled on Kaggle). Random seeds are fixed at 42, but GPU results can still vary slightly between runs.

## Repository layout

```
.
├── README.md
├── week6-steel-defect-classification.ipynb   # the Kaggle notebook with outputs
└── images/
    ├── sample_grid.png
    ├── closest_pairs.png
    ├── confusion_matrix.png
    └── gradcam.png
```
