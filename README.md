# multilingual-LM

## Overall
This repository is based on [mission-impossible-language-models](https://github.com/jkallini/mission-impossible-language-models).


## Setup and Environment
First, please clone the two repositories:
```commandline
git clone https://github.com/xiulinyang/multilingual-LM.git
cd multilingual-LM
rm -rf mistral
git clone https://github.com/xiulinyang/mistral.git
```
Second, please create two virtual environment by
```commandline
conda create -n mission python=3.9
conda activate mission
pip install -r requirements.txt
```
and 
```commandline

cd mistral
conda create -n mistral python=3.8.12 pytorch=1.11.0 torchdata cudatoolkit=11.3 -c pytorch
conda activate mistral
pip install -r setup/pip-requirements.txt
```

## Chanage the file paths

Please change the file or root paths in the following scripts:
- utils.py
- training/conf/template/gpt2-small-template.yaml
- training/conf/template/multilingual_dataset_template.yaml
- training/conf/template/multilingual_train_template.yaml (wandb)


## Run the experiments
You can simply run the experiment by the following command
```commandline
bash run.sh LANG GPU PERTURBATION RANDOM_SEED
#e.g., bash run.sh NL 0 shuffle_deterministic21 41
```
Parameters:
- LANG: Language code (e.g., EN, DE, etc.).
- GPU: The GPU ID to use.
- PERTURBATION: Perturbation type (defined in the FUNCTION_MAPS in utils.py).
- RANDOM_SEED: Random seed (in our experiments, we use 41, 53, 81).
