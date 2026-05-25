# Optimus environments

Three conda environments cover the full Optimus pipeline. Each `requirements-*.txt` lists **pip** dependencies; install **PyTorch + CUDA** via conda first (commands are in each file header).

## Environment matrix

| Conda env | Requirements file | Primary modules | Paper stage |
|---|---|---|---|
| **`llm_clone`** | [requirements-llm_clone.txt](requirements-llm_clone.txt) | `Data_processor/`, `LM_Toxic_detect/`, `Healing_creation/`, `Injection_code/`, `Evaluation/`, baselines (`toxic-prompt/`, `star-dss/`) | 0–4 (main stack) |
| **`chatbot_adv_deep`** | [requirements-chatbot_adv_deep.txt](requirements-chatbot_adv_deep.txt) | `PromptAttack/`, `AmpleGCG/`, `Universal-Prompt-Injection/` | §7 attacks |
| **`grade_env1`** | [requirements-grade_env1.txt](requirements-grade_env1.txt) | GRADE metric via `Injection_code/tools/GRADE.sh` | 4 (GRADE) |

## Quick setup

```bash
cd Optimus/environments

# Core pipeline + evaluation + baselines
conda create -n llm_clone python=3.10 -y
conda activate llm_clone
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-llm_clone.txt

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

## Notes

- **`llm_clone`** is the default environment for Stages 0–4, RTR evaluation, and comparison baselines.
- **`grade_env1`** uses Python 3.6 and TensorFlow 1.14 (GRADE upstream dependency). Keep it separate from `llm_clone`.
- API scorers in `LM_Toxic_detect/api_scorers/` need credentials via environment variables (Perspective, OpenAI, Azure Content Safety). See [LM_Toxic_detect/README.md](../LM_Toxic_detect/README.md).
