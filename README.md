Manuscript Region Detection



A Python-based computer vision pipeline for detecting and classifying layout regions in historical manuscript images.



The system uses a YOLO object-detection model to identify five layout categories:



header — top-margin text, running headers, section titles, and folio-like regions

footer — bottom-margin text, catchwords, page numbers, and signatures

main\_text — primary manuscript body text

side\_text — marginalia, annotations, and text positioned along page margins

filler — decorative elements and other non-main-text content



The pipeline supports batch processing and produces both annotated images and JSON prediction files.



1\. Project Features

Accepts a single image or an entire folder of images

Automatically processes supported manuscript images

Detects five manuscript layout classes

Produces bounding boxes around detected regions

Provides confidence scores for every prediction

Keeps bounding boxes inside image boundaries

Generates annotated output images

Generates JSON metadata for every processed image

Supports batch inference

Uses relative project paths

Does not require a database

Does not require a frontend

Uses a modular Python structure

2\. Project Structure

manuscript-region-detection/

│

├── data/

│   ├── images/

│   │   ├── train/

│   │   ├── val/

│   │   └── test/

│   │

│   └── labels/

│       ├── train/

│       ├── val/

│       └── test/

│

├── dataset/

│   └── extracted/

│       └── Dataset/

│

├── models/

│   └── best.pt

│

├── results/

│   ├── \*.json

│   └── \*\_result.jpg

│

├── runs/

│

├── src/

│   ├── detector.py

│   ├── postprocessing.py

│   └── visualization.py

│

├── inference.py

├── train.py

├── prepare\_dataset.py

├── inspect\_dataset.py

├── data.yaml

├── requirements.txt

├── yolo11n.pt

└── README.md

3\. Technology Stack

Python

Ultralytics YOLO

OpenCV

NumPy

PyTorch

JSON

Pillow



The object-detection model is based on the YOLO11 Nano architecture because it provides a lightweight model suitable for experimentation and CPU-based development while supporting object localization and classification.



4\. Dataset



The project uses historical manuscript images with corresponding ground-truth semantic segmentation masks.



The original dataset contains semantic labels such as:



Original Label	RGB Color	Meaning

Background	(0, 0, 0)	Background

Decoration	(255, 0, 0)	Decoration

Filler	(0, 255, 0)	Filler

Text	(0, 0, 255)	Text

Main Body	(255, 255, 0)	Main manuscript body

Drop Caps	(0, 255, 255)	Drop caps



The assignment requires five object-detection classes:



0 - header

1 - footer

2 - main\_text

3 - side\_text

4 - filler



Because the original dataset does not directly provide these exact five object-detection classes, the project uses a derived-label conversion process.



Label conversion

Main Body → main\_text

Filler → filler

Decoration → filler

Drop Caps → filler

Text → classified using page-relative geometric rules into:

header

footer

side\_text

main\_text



The header, footer, and side\_text labels generated from the original Text category are therefore derived pseudo-labels, not original annotations supplied by the dataset.



This distinction is important when interpreting evaluation results.



5\. Dataset Preparation



The preparation pipeline:



Matches manuscript images with their corresponding ground-truth masks.

Reads the semantic mask colors.

Converts the semantic regions into bounding boxes.

Applies the five assignment-specific classes.

Normalizes bounding boxes into YOLO format.

Clips bounding boxes to image boundaries.

Creates training and validation splits.

Copies the test images into the test directory.



The original dataset files are not modified.



The locally available matched training images are split into:



Train: 424 images

Validation: 107 images

Test: 300 images



The local dataset copy does not contain the complete train/validation image set described in the original dataset documentation, so the pipeline uses the matched images that are actually available locally.



6\. YOLO Dataset Configuration



The dataset is configured using data.yaml.



path: ./data



train: images/train

val: images/val

test: images/test



names:

&#x20; 0: header

&#x20; 1: footer

&#x20; 2: main\_text

&#x20; 3: side\_text

&#x20; 4: filler

7\. Installation



Create and activate a virtual environment:



python -m venv .venv



Activate it on Windows PowerShell:



.\\.venv\\Scripts\\Activate.ps1



Install the dependencies:



pip install -r requirements.txt



Verify Ultralytics:



pip show ultralytics

8\. Training



The training script is provided in:



train.py



A trained model is stored as:



models/best.pt



Training was performed using the prepared five-class dataset.



The model was evaluated using standard object-detection metrics including:



Precision

Recall

mAP@50

mAP@50–95



The best recorded validation result in the current training run was:



Precision: 0.83236

Recall:    0.64420

mAP@50:    0.65931

mAP@50–95: 0.51227



These values describe the current training run and should not be interpreted as class-specific performance.



9\. Inference



The main inference entry point is:



inference.py

Single image

python inference.py --input ./data/images/test/example.jpg --output ./results

Folder of images

python inference.py --input ./data/images/test --output ./results

Using a different model

python inference.py --input ./data/images/test --output ./results --model ./models/best.pt



The assignment-style command is:



python inference.py --input ./data/test\_images --output ./results



provided that the input directory exists.



10\. Output



For every input image, the pipeline creates:



An annotated image

A JSON prediction file



Example:



results/

├── manuscript\_001\_result.jpg

├── manuscript\_001.json

├── manuscript\_002\_result.jpg

├── manuscript\_002.json

└── ...

11\. JSON Format



Example prediction:



{

&#x20;   "image": "example.jpg",

&#x20;   "predictions": \[

&#x20;       {

&#x20;           "label": "main\_text",

&#x20;           "class\_id": 2,

&#x20;           "confidence": 0.7503,

&#x20;           "bbox": {

&#x20;               "x1": 175,

&#x20;               "y1": 137,

&#x20;               "x2": 507,

&#x20;               "y2": 597

&#x20;           }

&#x20;       }

&#x20;   ]

}



Each prediction contains:



label — detected region class

class\_id — numeric YOLO class ID

confidence — model confidence

bbox — bounding-box coordinates

12\. Bounding Box Handling



Bounding boxes are represented using:



x1, y1, x2, y2



where:



x1 = left coordinate

y1 = top coordinate

x2 = right coordinate

y2 = bottom coordinate



The post-processing module ensures that coordinates remain inside the image boundaries.



13\. Modular Components

src/detector.py



Responsible for:



Loading the trained YOLO model

Running object detection

Processing model predictions

Producing detection results

src/postprocessing.py



Responsible for:



Cleaning detection results

Clipping bounding boxes

Formatting predictions

src/visualization.py



Responsible for:



Drawing bounding boxes

Displaying class labels

Displaying confidence values

Creating annotated images

inference.py



Provides the command-line interface for running inference on individual images or folders.



14\. Robustness Considerations



Historical manuscript images may contain:



Faded ink

Uneven illumination

Blur

Skew

Stains

Bleed-through

Page damage

Different aspect ratios

Multiple text columns

Different handwriting or writing styles



The model is trained on varied manuscript imagery and uses image augmentation available through the YOLO training pipeline.



The system is intended as a practical detection pipeline rather than a guarantee of perfect detection for every manuscript condition.



15\. Current Evaluation Notes



The current model demonstrates useful overall object-detection performance.



However, the derived dataset is imbalanced between classes. In particular, side\_text and footer have substantially fewer generated training examples than main\_text.



Therefore, overall mAP should not be treated as proof that every required class performs equally well.



Class-specific evaluation and additional training may be required if marginal text or footer regions are a major evaluation criterion.



16\. Reproducibility



The project uses relative paths so that it can be moved between machines.



The main execution steps are:



python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt



Prepare the dataset if required:



python prepare\_dataset.py



Train the model:



python train.py



Run inference:



python inference.py --input ./data/images/test --output ./results

17\. Design Decisions

Why YOLO?



YOLO was selected because it:



Performs object detection and classification together

Produces bounding boxes directly

Supports batch inference

Is relatively lightweight

Provides confidence scores

Is practical for a command-line computer-vision pipeline

Why a lightweight model?



The development environment is CPU-based. A lightweight YOLO model reduces training and inference requirements while providing a practical baseline for the assignment.



Why JSON?



JSON is used for prediction metadata because it is:



Lightweight

Human-readable

Easy to process programmatically

Suitable for storing bounding boxes, labels, and confidence values



No database is required for this assignment.



18\. Limitations



The main limitation is the difference between the original dataset's semantic segmentation categories and the assignment's required five object-detection categories.



In particular, the original Text category does not explicitly distinguish:



Header

Footer

Main text

Side text



These categories are therefore derived using spatial rules.



As a result, some class-specific errors may occur, especially for marginal text, footer text, or layouts where the spatial position is ambiguous.



Future improvement can include manually verified annotations for the five target classes and additional class-balanced training.



19\. License and Dataset Attribution



The original dataset's license and attribution requirements should be preserved according to the dataset documentation.



The original input images and metadata are not modified by the preparation pipeline.



20\. Summary



This project provides an end-to-end manuscript layout detection pipeline:



Manuscript Image

&#x20;      ↓

Dataset Preparation

&#x20;      ↓

Derived YOLO Labels

&#x20;      ↓

YOLO Training

&#x20;      ↓

Trained Model

&#x20;      ↓

CLI Inference

&#x20;      ↓

Region Detection

&#x20;      ↓

┌───────────────────────────┐

│ header                    │

│ footer                    │

│ main\_text                 │

│ side\_text                 │

│ filler                    │

└───────────────────────────┘

&#x20;      ↓

Annotated Image + JSON



The implementation is designed to be modular, reproducible, batch-capable, and suitable for command-line execution.

