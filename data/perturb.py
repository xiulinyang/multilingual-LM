# perturb.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from utils import EXP_LANGS, MULTILINGUAL_SPLITS, MULTILINGUAL_DATA_PATH,\
    GENRES, write_file, MARKER_TOKEN_IDS, marker_rev_token, marker_sg_token, marker_pl_token, get_perturbations, \
    FUNCTION_MAP
from glob import glob
import numpy as np
import itertools
import json
import os
import tqdm
import argparse
import pytest

def lines_equivalent_3pres(file1_path, file2_path):
    """Compare lines of two files after splitting them."""
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            # Split each line and compare the resulting lists
            res1 = [i for i in line1.split() if int(
                i) not in (marker_sg_token, marker_pl_token)]
            res2 = [i for i in line2.split() if int(
                i) not in (marker_sg_token, marker_pl_token)]
            if res1 != res2:
                print(line1)
                print(line2)
                return False

        # Check if one file has more lines than the other
        if file1.readline() or file2.readline():
            return False

    return True


perturbation_pairs_3pres = [
    ("0tokens", "4tokens"),
    ("0tokens", "4words"),
    ("4tokens", "4words"),
]
test_data = itertools.product(
    ["EN", "dev", "test_affected", "test_unaffected"], GENRES.keys(), perturbation_pairs_3pres)


@pytest.mark.parametrize("split, genre, perturbation_pair", test_data)
def test_3pres_all_equivalent(split, genre, perturbation_pair):

    perturbation1, perturbation2 = perturbation_pair

    if split in EXP_LANGS:
        filename = f"{genre}.train"
    elif split == "test_affected":
        filename = f"{genre}_affected.test"
    elif split == "test_unaffected":
        filename = f"{genre}_unaffected.test"
    elif split == "dev":
        filename = f"{genre}.dev"

    path1 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_3pres_{perturbation1}/{lang}/{split}/{filename}"
    path2 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_3pres_{perturbation2}/{lang}/{split}/{filename}"

    assert lines_equivalent_3pres(path1, path2), f"File {filename} of " + \
        f"3pres_{perturbation1} and 3pres_{perturbation2} have non-equivalent lines!"


def lines_equivalent_reversal(rev_path, ident_path):
    """Compare lines of reversal file and identity file after splitting them."""
    with open(rev_path, 'r') as file1, open(ident_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            # Split each line and compare the resulting lists
            line1_tokens = line1.split()
            line2_tokens = line2.split()

            # Get REV marker index
            marker_index = line1_tokens.index(str(marker_rev_token))

            # Make sure tokens up to and including the marker are all the same
            if line1_tokens[:marker_index+1] != line2_tokens[:marker_index+1]:
                return False
        
            # Make sure reversal of rest of string is equal to identity
            line1_tokens_rev = line1_tokens[marker_index+1:].copy()
            line1_tokens_rev.reverse()
            if line1_tokens_rev != line2_tokens[marker_index+1:]:
                return False

        # Check if one file has more lines than the other
        if file1.readline() or file2.readline():
            return False

    return True
        

perturbation_pairs_reversal = [
    ("reversal", "reversal_identity"),
]
test_data = itertools.product(
    ["EN",  "dev", "test_affected"], GENRES.keys(), perturbation_pairs_reversal)

@pytest.mark.parametrize("split, genre, perturbation_pair", test_data)
def test_reversal_all_equivalent(split, genre, perturbation_pair):

    perturbation1, perturbation2 = perturbation_pair

    if split in ['train']:
        filename = f"{genre}.train"
    elif split == "test_affected":
        filename = f"{genre}_affected.test"
    elif split == "test_unaffected":
        filename = f"{genre}_unaffected.test"
    elif split == "dev":
        filename = f"{genre}.dev"

    path1 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_{perturbation1}/multilingual_{split}/{filename}"
    path2 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_{perturbation2}/multilingual_{split}/{filename}"

    assert lines_equivalent_reversal(path1, path2), f"File {filename} of " + \
        f"{perturbation1} and {perturbation2} have non-equivalent lines!"


def lines_equivalent_determiner_swap(det_path, ident_path):
    """Compare lines of reversal file and identity file after splitting them."""
    with open(det_path, 'r') as file1, open(ident_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            # Split each line and compare the resulting lists
            line1_tokens = set(line1.split())
            line2_tokens = set(line2.split())
            if line1_tokens != line2_tokens:
                print(line1.split())
                print(line2.split())
                return False

        # Check if one file has more lines than the other
        if file1.readline() or file2.readline():
            return False

    return True


perturbation_pairs_reversal = [
    ("determiner_swap", "determiner_swap_identity"),
]
test_data = itertools.product(
    [ "dev", "test_affected", "test_unaffected"], GENRES.keys(), perturbation_pairs_reversal)

@pytest.mark.parametrize("split, genre, perturbation_pair", test_data)
def test_determiner_swap_all_equivalent(split, genre, perturbation_pair):

    perturbation1, perturbation2 = perturbation_pair

    if split in EXP_LANGS:
        filename = f"{genre}.train"
    elif split == "test_affected":
        filename = f"{genre}_affected.test"
    elif split == "test_unaffected":
        filename = f"{genre}_unaffected.test"
    elif split == "dev":
        filename = f"{genre}.dev"

    path1 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_{perturbation1}/{lang}/{split}/{filename}"
    path2 = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/multilingual_{perturbation2}/{lang}/{split}/{filename}"

    assert lines_equivalent_determiner_swap(path1, path2), f"File {filename} of " + \
        f"{perturbation1} and {perturbation2} have non-equivalent lines!"


def flatten_list(l):
    """Function to flatten a nested list."""
    return list(itertools.chain.from_iterable(l))


def process_line(line,lang):
    """
    Process a given line from the dataset, apply transformations to its sentences,
    and categorize them into affected or unaffected based on the transformation.

    Parameters:
    - line (dict): A dictionary representing a line from the dataset, which contains
      sentence annotations.

    Returns:
    - tuple: A tuple containing three lists:
        1. new_lines_affected (list of str): Sentences that were affected by the transformation.
        2. new_lines_unaffected (list of str): Sentences that were not affected by the transformation.

    Note:
    - The transformation functions (`perturbation_function`, `affect_function`, `filter_function`)
      are expected to be available in the global scope.
    """

    new_lines_affected = []
    new_lines_unaffected = []
    sents_unaffected = []

    # Apply transformation to each sentence on line
    for sent in line["sent_annotations"]:

        tokens = perturbation_function(sent)
        if len([tok for tok in tokens if tok not in MARKER_TOKEN_IDS]) <= 1:
            continue

        token_line = " ".join([str(tok) for tok in tokens])

        # Check if sent is affected
        if affect_function(sent,lang):

            # Check if this affected sentence should be filtered or not
            if filter_function(sent,lang):
                new_lines_affected.append(token_line + "\n")

        else:  # Unaffected sentences
            new_lines_unaffected.append(token_line + "\n")
            sents_unaffected.append(sent["sent_text"] + "\n")

    return new_lines_affected, new_lines_unaffected, sents_unaffected


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Perturb MULTILINGUAL dataset',
        description='Perturb MULTILINGUAL dataset by altering POS-tagged data')
    parser.add_argument('perturbation_type',
                        type = str,
                        default='all',
                        const='all',
                        nargs='?',
                        choices=FUNCTION_MAP.keys(),
                        help='Perturbation function used to transform MULTILINGUAL dataset')
    parser.add_argument('lang',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=EXP_LANGS,
                        help='MULTILINGUAL dataset choice')

    parser.add_argument('split',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=['train', 'dev', 'test'],
                        help='MULTILINGUAL dataset choice')
    # Get args
    args = parser.parse_args()

    # Load dataset (only json files containing tagged data)
    multilingual_dataset = args.lang
    multilingual_split = args.split
    json_ext = ".json"
    language_lower = args.lang.lower()
    PERTURBATIONS = get_perturbations(args.lang, args.perturbation_type)

    multilingual_data = glob(f"{MULTILINGUAL_DATA_PATH}/{multilingual_dataset}/{multilingual_split}/*{json_ext}")
    print(f"{MULTILINGUAL_DATA_PATH}/{multilingual_dataset}/{multilingual_split}/*{json_ext}")
    # Get perturbation, affect, and filter functions
    perturbation_function = PERTURBATIONS[f'{args.perturbation_type}_{language_lower}']['perturbation_function']
    affect_function = PERTURBATIONS[f'{args.perturbation_type}_{language_lower}']['affect_function']
    filter_function = PERTURBATIONS[f'{args.perturbation_type}_{language_lower}']['filter_function']
    gpt2_tokenizer = PERTURBATIONS[f'{args.perturbation_type}_{language_lower}']['gpt2_tokenizer']
    lang = args.lang
    print(gpt2_tokenizer)
    print(f'token id of end of text: ')
    print(gpt2_tokenizer.encode('<|endoftext|>'))
    print(len(gpt2_tokenizer))
    if multilingual_split == "test":

        # Iterate over files and do transform
        for file in multilingual_data:
            print(file)
            f = open(file)
            data = json.load(f)
            f.close()

            # Perturb data iteratively
            results = []
            for line in tqdm.tqdm(data):
                results.append(process_line(line,lang))

            new_lines_affected, new_lines_unaffected, unaffected_sents = zip(
                *results)
            new_lines_affected = flatten_list(new_lines_affected)
            new_lines_unaffected = flatten_list(new_lines_unaffected)
            unaffected_sents = flatten_list(unaffected_sents)

            # Name new file
            new_file_affected = os.path.basename(
                file).replace(json_ext, "_affected.test")
            new_file_unaffected = os.path.basename(
                file).replace(json_ext, "_unaffected.test")
            file_unaffected_sents = os.path.basename(
                file).replace(json_ext, "_unaffected_sents.test")

            # Create directory
            data_write_directory = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed"
            directory_affected = f"{data_write_directory}/{args.perturbation_type}_{language_lower}/test_affected/"
            if not os.path.exists(directory_affected):
                os.makedirs(directory_affected)
            directory_unaffected = f"{data_write_directory}/{args.perturbation_type}_{language_lower}/test_unaffected/"
            if not os.path.exists(directory_unaffected):
                os.makedirs(directory_unaffected)
            directory_unaffected_sents = f"{data_write_directory}/{args.perturbation_type}_{language_lower}/test_unaffected_sents/"
            if not os.path.exists(directory_unaffected_sents):
                os.makedirs(directory_unaffected_sents)

            # Write files
            write_file(directory_affected,
                       new_file_affected, new_lines_affected)
            write_file(directory_unaffected,
                       new_file_unaffected, new_lines_unaffected)
            write_file(directory_unaffected_sents,
                       file_unaffected_sents, unaffected_sents)

    else:
        # Iterate over files and do transform
        for file in multilingual_data:
            print(file)
            f = open(file)
            data = json.load(f)
            f.close()

            # Perturb data iteratively
            results = []
            for line in tqdm.tqdm(data):
                results.append(process_line(line,lang))

            new_lines_affected, new_lines_unaffected, _ = zip(
                *results)

            new_lines_affected = flatten_list(new_lines_affected)
            new_lines_unaffected = flatten_list(new_lines_unaffected)

            # Combine affected and unaffected sentences
            new_lines = new_lines_unaffected + new_lines_affected

            # Name new file
            if multilingual_split == "dev":
                new_file = os.path.basename(file).replace(json_ext, ".dev")
            elif multilingual_split == 'unittest':
                new_file = os.path.basename(file).replace(json_ext, ".test")

                # Print strings for unittest
                new_lines_decoded = [gpt2_tokenizer.decode(
                    [int(tok) for tok in line.split()])[:1024] + "\n" for line in new_lines]
                new_lines_with_strings = []
                for tokens, line in list(zip(new_lines, new_lines_decoded)):
                    new_lines_with_strings.append(tokens)
                    new_lines_with_strings.append(line)
                new_lines = new_lines_with_strings

            else:
                new_file = os.path.basename(file).replace(json_ext, ".train")

            # Create directory and write file
            directory = f"{MULTILINGUAL_DATA_PATH}/multilingual_data_perturbed/{args.perturbation_type}_{language_lower}/{multilingual_split}/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            write_file(directory, new_file, new_lines)
