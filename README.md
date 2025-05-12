# Learning Never Stops: Improving Software Vulnerability Type Identification via Incremental Learning

This is the source code to the paper "Learning Never Stops: Improving Software Vulnerability Type Identification via Incremental Learning". Please refer to the paper for the experimental details.

## Approach
<img src="figs/framework.png">

## Datasets
Our dataset is licensed under the GPL 3.0, as found in the [LICENSE](datasets/LICENSE.txt) file.

The processed dataset you can download from <a href="https://drive.google.com/drive/folders/1GuchdeFsGUKh8tvCles9kcjIcC-loD5v?usp=drive_link">Google Drive</a>.

## Requriements
You can install the required dependency packages for our environment by using the following command: pip install - r requirements.txt.

## Data preprocess
Simulate our dataset into a continuous stream. Like we split our original dataset into five tasks.

## Reproducing the experiments:
1.Use the py file under ``data crawling and processing`` for data processing. Of course, you can directly use the ``dataset`` we have processed: [Google Drive Link](https://drive.google.com/drive/folders/1P42XsDWeMqAW33oS0gGamXEqxYiMjO5i?usp=drive_link)

2.Run ``VulTypeIL.py``. After running, you can retrain the ``model`` and obtain results.

3.You can find the implementation code for the ``RQ1-RQ5`` section and the ``Discussion`` section experiments in the corresponding folders. 

## Pre-trained model
You can obtain our saved model and reproduce our results through the <a href="https://drive.google.com/drive/folders/1GuchdeFsGUKh8tvCles9kcjIcC-loD5v">model link</a>
