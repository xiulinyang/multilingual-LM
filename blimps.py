# perplexities.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from transformers import GPT2LMHeadModel
from utils import CHECKPOINT_READ_PATH, FUNCTION_MAP, MULTILINGUAL_DATA_PATH, TOKENIZER, EXP_LANGS
from tqdm import tqdm
from glob import glob
from numpy.random import default_rng
import pandas as pd
import torch
import itertools
import argparse
import os



MAX_TRAINING_STEPS = 1200
CHECKPOINTS = list(range(0, MAX_TRAINING_STEPS+1, 100))


def create_attention_mask(token_lists):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    mask = torch.full((batch_size, seq_length), 0)

    for i, tokens in enumerate(token_lists):
        mask[i, 0:len(tokens)] = 1

    return mask


def create_input_ids(lang, token_lists, pad_token_id):
    if lang=='ZH':
        pad_token_id = 0
    padded = zip(*itertools.zip_longest(*token_lists, fillvalue=pad_token_id))
    return torch.tensor(list(padded))


def get_perplexities(model, token_lists, pad_token_id, lang, device="cuda"):

    # Prepare data
    input_ids = create_input_ids(lang, token_lists, pad_token_id).to(device)
    labels = input_ids.clone()  # GPT-2 uses input as labels for CLM task
    attention_mask = create_attention_mask(token_lists).to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask)

    # The "shifted" nature of labels in GPT-2 (next token prediction)
    # Shift logits, labels, and attention mask by one position
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Instantiate loss function with no reduction
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Calculate per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    # Reshape back to the original batch size and sequence length
    loss = loss.view(shift_labels.size())

    # Apply the attention mask - only calculate loss where mask is 1
    loss = loss * shift_attention_mask

    # Sum the loss over the sequence length, get per-example perplexity
    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    return torch.exp(per_example_loss).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Edge probing',
        description='Edge probing experiments')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=FUNCTION_MAP.keys(),
                        help='Perturbation function used to transform the multilingual dataset')
    parser.add_argument('test_perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=FUNCTION_MAP.keys(),
                        help='Perturbation function used to transform test the multilingual dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=EXP_LANGS,
                        help='BabyLM train set')
    parser.add_argument('random_seed', type=int, help="Random seed")
    parser.add_argument('paren_model',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["randinit"],
                        help='Parenthesis model')
    # Get args
    args = parser.parse_args()
    la = args.train_set
    lang_lower_case = args.train_set.lower()
    gpt2_tokenizer = TOKENIZER[la]['shuffle']
    # Get path to model
    model = f"{args.perturbation_type}_{lang_lower_case}_{args.train_set}_{args.paren_model}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_READ_PATH}/{args.perturbation_type}_{lang_lower_case}_{args.train_set}_{args.paren_model}/babylm_{model}/runs/{model}/checkpoint-"

    # Get perturbed test files
    test_files = sorted(glob(
        f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/{args.test_perturbation_type}_{lang_lower_case}/test_affected/*"))

    # Iterate over data files to get perplexity data
    print("Sampling BabyLM affected test files to extract surprisals...")
    token_sequences = []
    categories=[]
    for test_file in test_files:
        print(test_file)
        category = str(test_file.split('/')[-1])
        categories.append(category)
        # Get tokens from test file and subsample
        f = open(test_file, 'r')
        file_token_sequences = [
            [int(s) for s in l.split()] for l in f.readlines()]
        sample_indices = list(range(len(file_token_sequences)))

        file_token_sequences = [file_token_sequences[i]
                                for i in sample_indices]
        token_sequences.extend(file_token_sequences)

    # For logging/debugging, include decoded sentence
    test_sents = [gpt2_tokenizer.decode(
        toks) for toks in token_sequences]

    ppl_df = pd.DataFrame({
        "Sentences": test_sents,
        "Categories": categories
    })

    BATCH_SIZE = 8
    device = "cuda"
    for ckpt in CHECKPOINTS:
        print(f"Checkpoint: {ckpt}")

        # Load model
        model = GPT2LMHeadModel.from_pretrained(
        model_path + str(ckpt)).to(device)

        # Get perplexities
        perplexities = []
        for i in tqdm(range(0, len(token_sequences), BATCH_SIZE)):
            batch = token_sequences[i:i+BATCH_SIZE]
            ppls = get_perplexities(
                model, batch, gpt2_tokenizer.eos_token_id, la)
            perplexities.extend(ppls)

        # Add ppls to df
        ppl_df[f'Perplexities (ckpt {ckpt})'] = perplexities

    directory = f"blimps_results/{args.perturbation_type}_{args.train_set}"
    file = directory + \
           f"/{args.paren_model}_seed{args.random_seed}_{args.perturbation_type}.csv"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Writing results to CSV: {file}")
    ppl_df.to_csv(file)