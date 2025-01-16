# multilingual-LM

## Overall
This repository is based on [mission-impossible-language-models](https://github.com/jkallini/mission-impossible-language-models).

# Condor submit
Use the submit/run.sub to submit. Replace \<username\>, \<projdir\>, \<envname\> and \<wandbkey\> in all files under submit/ with correct names.

# TODO List

## Experiment
- [ ] EN shuffle_local2 41|53|81
- [ ] AR shuffle_control|shuffle_local3|shuffle_local5|shuffle_local10|shuffle_deterministic21|shuffle_deterministic84|shuffle_deterministic57|shuffle_nondeterministic|shuffle_even_odd 41|53|81
- [ ] PL shuffle_control|shuffle_local3|shuffle_local5|shuffle_local10|shuffle_deterministic21|shuffle_deterministic84|shuffle_deterministic57|shuffle_nondeterministic|shuffle_even_odd 41|53|81
- [ ] FR shuffle_control|shuffle_local3|shuffle_local5|shuffle_local10|shuffle_deterministic21|shuffle_deterministic84|shuffle_deterministic57|shuffle_nondeterministic|shuffle_even_odd 41|53|81
- [ ] EN perturb_adj_num 41|53|81 (Xiulin needs to upload data!!)

## Assignments
- **Xiulin**
  - [ ] RU
  - [ ] DE
  - [ ] NL
- **Yuekun**
  - [ ] ZH
  - [ ] IT
  - [ ] RO

## STEP1: Setup and Environment
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

## STEP2: Change the file paths

Please change the file or root paths in the following scripts:
- utils.py
- training/conf/template/gpt2-small-template.yaml
- training/conf/template/multilingual_dataset_template.yaml
- training/conf/template/multilingual_train_template.yaml (wandb)

## STEP3: Prepare impossible languages
```commandline
cd data
python tag.py path/to/language/file -b batch_size -l LANG 
# e.g., python tag.py multilingual/RU/train/RU.train -b 2 -l RU 
# I usually set the batch size small because in my algorithm if stanza cannot parse the sentence, I will give up the whole batch
```

## STEP4: Run the experiments
You can simply run the experiment by the following command
```commandline
bash run.sh LANG PERTURBATION RANDOM_SEED
#e.g., bash run.sh NL shuffle_deterministic21 41
```
Parameters:
- LANG: Language code (e.g., EN, DE, etc.).
- PERTURBATION: Perturbation type (defined in the FUNCTION_MAPS in utils.py).
- RANDOM_SEED: Random seed (in our experiments, we use 41, 53, 81).
