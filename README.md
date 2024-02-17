# drugs-screening

Implementation of AI-driven method for CNS drug research utilizing adult zebrafish. The method is based on the novel tank test (NTT) behavioral experiment.

<div align="center">
  <img src="https://github.com/Lostbelt/drug-screening/blob/main/notebooks/Screenshot_2.png" width="500"/>
</div>

<div align="center">
  <img src="https://github.com/Lostbelt/drug-screening/blob/main/notebooks/Screenshot_1.png" width="300"/>
</div>

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
pip install torch torchvision scikit-image ultralytics scikit-learn
```

## Data preparation

First download [models checkpoints](https://drive.google.com/drive/folders/1Ahy9nWQRqqwMCV9Di8jhIZyxUQmo5NaC?usp=sharing) and place them in the weight folder.

If you need to further train the tank detector or the zebrafish detector, use the instructions at this [link](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format).

All your videos should be in one folder as shown below

```
data
├── video_1.mp4
├── video_2.mp4
├── video_3.mp4
├── video_4.mp4
├── ...
```


## <a name="GettingStarted"></a>Getting Started

Start video processing with the command
```
python run.py <video_files_dir>
```

Next, look at the processing file in the notebooks folder
