# multilingual-LM

## Overall
This repository is based on [mission-impossible-language-models](https://github.com/jkallini/mission-impossible-language-models).

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

## STEP3: Prepare language data for the experiments
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

# Experiment with other languages and other perturbations
If you want to experiment with additional languages or apply perturbations beyond those discussed in our paper, follow these steps:
## New langauges
1**Add your language data**:  
   Place the new language files in the `data/` folder, maintaining the existing data structure.
2. **Update language references**:  
   - Add the new language name to `util.py` and `training/multilingual_dataset.py`.  
   - Update the tokenizer configuration in `mistral/conf/models/`.  
## New perturbations
- Please update the `util.py` and `training/multilingual_dataset.py` with your new perturbation function. 

If you need the OPUS12 and OPUS30 corpus, or have any questions, feel free to open an issue or contact me at **xiulin.yang.compling@gmail.com**.