# Installation

## Requirements
1. To run large models like Blenderbot 400M, Here are the suggested minimum requirements:
    1. High-end NVIDIA GPUs with at least 48GB of RAM.
    2. Either Linux or Windows. We recommend Linux for better performance.
    3. CUDA Toolkit 11.4 , cudnn 8.5+, and the latest NVIDIA driver.

2. For smaller models like BART (DD-BART), Here are the suggested minimum requirements:
    1. High-end NVIDIA GPUs with at least 25GB of RAM.
    2. Either Linux or Windows. We recommend Linux for better performance.
    3. CUDA Toolkit 11.7 , cudnn 8.5+, and the latest NVIDIA driver.


## Clone the Project
To clone the project from github. Run the command below in the terminal.

```
git clone https://github.com/secml-lab-vt/Chatbot-Toxicity-Injection.git
cd chatbot-security-code
```

## Create and install Conda Environments

### Conda environment (chatbot) for the entire project (Only need to activate this conda environment)
```
# Create a conda environment chatbot using a YML file
conda env create -f dbl_env.yml

# Activate the conda environment
conda activate chatbot
```
### Conda environment for the GRADE metrics (This conda environment will be activated internally)
```
# Create a conda environment chatbot using a YML file
conda env create -f grade_env.yml

#Then install spacy model
conda install -c conda-forge spacy-model-en_core_web_md
```

# Model & Data Zoo




Here are list of datasets and Pretrained models below that can be downloaded from the Google Drive (open): [link](https://drive.google.com/drive/folders/1d6snNV6CJwzfEzlDd_xhxcXJ9x4BdtlI?usp=sharing)

|Models/Files which must be downloaded | |
| ----------- | ----------- |
|**Datasets --- (./datasets/datasets/)**| 
|[DailyDialog dataset](http://yanran.li/dailydialog) | Dataset to train DD-BART|
|PersonaChat train set | Dataset for sampling conversations|
|**Base models --- (./saves/base/)**| 
|DD-BART|BART model fine-tuned on DailyDialog |
|**Toxicity Classifiers --- (./saves/toxic_classifier/)**|
|WTC | Toxicity classifier trained on Wiki Toxic Comments|
|WTC-adv | Toxicity classfier trained on adversarial samples + WTC|


Here are list of models below that will be provided by filling a Google form:

|Models/Files which must be downloaded | |
| ----------- | ----------- |
|**Toxicity bots for attacks --- (./saves/toxic_bot/)**| 
|TBot| Toxic bot trained regularly on Reddit text|
|TBot-S| Toxic bot trained for more repetitions|

### Google form for datasets and model requests:
In order to download the TBot and Tbot-S models mentioned below, please fill out the Google form [link](https://forms.gle/LUZ8CHNZK4mhwmsM6) after reading and agreeing our License Agreement. Upon acceptance of your request, the download link will be sent to the provided e-mail address.


### We have used source code from various works. We thank the authors for their open source contribution to promote research
* [GRADE](https://github.com/li3cmz/GRADE)
* [GRUEN](https://github.com/WanzhengZhu/GRUEN)
* [Text Fooler](https://github.com/jind11/TextFooler)
* [Hugging Face](https://huggingface.co/)


## GRADE/GRUEN/TextFooler setup
1. For GRADE, follow the instructions at [https://github.com/li3cmz/GRADE]() to download the pretrained GRADE model. 

2. For TextFooler, follow the instructions at [https://github.com/jind11/TextFooler]() to download the counter-fitting word embeddings. 

3. For GRUEN, no additional steps are needed for installation

# Folder Path Structure
Cached conversations and dialog datasets can be found in [data/](). Toxicity and quality results can be found in [results/](). DBL model checkpoints are trained in [saves/]().