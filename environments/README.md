# Optimus environments

## Environment matrix

| Conda env | Requirements file |
|---|---|
| **`chatbot_adv_deep`** | [requirements-chatbot_adv_deep.txt](requirements-chatbot_adv_deep.txt)
| **`grade_env1`** | [requirements-grade_env1.txt](requirements-grade_env1.txt) 

## Quick setup

```bash
cd Optimus/environments

# Adaptive attacks (§7)
conda create -n chatbot_adv_deep python=3.10 -y
conda activate chatbot_adv_deep
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-chatbot_adv_deep.txt

# GRADE utility metric (isolated legacy stack)
conda create -n grade_env1 python=3.6.13 -y
conda activate grade_env1
pip install -r requirements-grade_env1.txt
```