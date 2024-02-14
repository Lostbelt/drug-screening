# drugs-screening

Implementation of AI-driven method for CNS drug research utilizing adult zebrafish. The method is based on the novel tank test (NTT) behavioral experiment.

## Installation

Create a conda environment and activate it

```
conda create --name screening python=3.9 -y
conda activate screening
```

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```


Install dependencies

```
git clone https://github.com/Lostbelt/drug-screening.git
cd drug-screening
pip install ultralytics torch torchvision scikit-image
```

## <a name="GettingStarted"></a>Getting Started

First download [models checkpoints](https://drive.google.com/drive/folders/1Ahy9nWQRqqwMCV9Di8jhIZyxUQmo5NaC?usp=sharing) and place it to Checkpoints.

Start video processing with the command
```
python run.py <video_files_dir>
```
