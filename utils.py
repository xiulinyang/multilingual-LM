# utils.py
# Author: Julie Kallini

from collections import deque
from string import punctuation
from warnings import WarningMessage

from transformers import AutoTokenizer, AddedToken,GPT2LMHeadModel, GPT2Tokenizer,AutoTokenizer, AutoModel
from functools import partial
from numpy.random import default_rng
from nltk.tree import ParentedTree
import torch


##############################################################################
# CONSTANTS
##############################################################################
ROOT_PATH = '/local/xiulyang'
EXP_LANGS = ['EN', 'DE', 'AR', 'ZH', 'RU', 'TR', 'RO','ES', 'FR', 'PL', 'PT', 'NL', 'IT', 'FR']
BABYLM_SPLITS = ["train", 'dev', 'test', 'unittest']
SEEDS = [21, 53, 84]
CHECKPOINTS = list(range(50, 501, 50))
GENRES = {
    "aochildes": "CHILDES",
    "bnc_spoken": "British National Corpus (BNC)",
    "cbt": "Children’s Book Test",
    "children_stories": "Children’s Stories Text Corpus",
    "gutenberg": "Standardized Project Gutenberg Corpus",
    "open_subtitles": "OpenSubtitles",
    "qed": "QCRI Educational Domain Corpus",
    "simple_wikipedia": "Simple Wikipedia",
    "switchboard": "Switchboard Dialog Act Corpus",
    "wikipedia": "Wikipedia"
}
CHECKPOINT_WRITE_PATH = f"/{ROOT_PATH}/babylm_models"
CHECKPOINT_READ_PATH = f"/{ROOT_PATH}/babylm_models"
TOKENIZER_PATH = f"/{ROOT_PATH}/mission-impossible-language-models/tokenizers"
BABYLM_DATA_PATH = f"/{ROOT_PATH}/mission-impossible-language-models/data/multilingual/"
MARKER_HOP_SING = "🅂"
MARKER_HOP_PLUR = "🄿"
MARKER_REV = "🅁"
BOS_TOKEN = "<BOS_TOKEN>"
PART_TOKENS = set(["n't", "'ll", "'s", "'re", "'ve", "'m"])
PUNCT_TOKENS = set(punctuation)


##############################################################################
# PARENS MODELS (Structurally-pretrained)
##############################################################################


PAREN_MODEL_PATH = "/u/scr/isabelvp//tilt-stuff/tilt-finetuning/pretrained_checkpoints/"
PAREN_MODELS = {
    "CROSS": "flat-parens_vocab500-uniform_deplength-nesting-nolimit",
    "NEST": "nested-parens0.49_vocab500-uniform",
    "RAND": "random_vocab500-uniform",
}


##############################################################################
# HELPER FUNCTIONS
##############################################################################


def write_file(directory, filename, lines):
    f = open(directory + filename, "w")
    f.writelines(lines)
    f.close()


def get_gpt2_tokenizer_with_markers(marker_list, lang):
    if lang in EXP_LANGS:    
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DICT[lang])
    else:
        WarningMessage("You didn't specify a language yet, "
                       "so we use the English gpt2 tokenizer by default!")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # If no new markers to add, return normal tokenizer
    if len(marker_list) == 0:
        return tokenizer

    # Create tokens and return modified tokenizer
    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer



# MARKER_TOKEN_IDS = [marker_sg_token, marker_pl_token, marker_rev_token]
MARKER_TOKEN_IDS =[]
def compute_surprisals(model, input_ids):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        shifted_input_ids = input_ids[:, 1:]

        # Get the log probabilities for the actual next tokens
        log_probs = torch.log2(torch.nn.functional.softmax(logits, dim=-1))
        true_log_probs = log_probs.gather(
            2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Get the negative log probabilities
    neg_log_probs = (-true_log_probs).tolist()
    surprisals = [[None] + probs for probs in neg_log_probs]
    return surprisals


def compute_token_probabilities(model, input_ids, token_id, pad_token_id):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probabilities for the specified token at each position
        token_probs = probs[:, :, token_id]

    # Convert to list and add None at the beginning to align with input tokens
    # Put null probability for instances of pad token
    token_probs_list = []
    for batch_i, probs in enumerate(token_probs):
        input_ids_seq = input_ids[batch_i].tolist() + [pad_token_id]
        filtered = [p if input_ids_seq[pos_i+1] !=
                    pad_token_id else None for pos_i, p in enumerate(probs.tolist())]
        token_probs_list.append([None] + filtered)

    return token_probs_list


def merge_part_tokens(words):
    result = []
    for s in words:
        if result and s in PART_TOKENS and len(result) > 0:
            result[-1] += s
        else:
            result.append(s)
    return result


def __affect_hop_word(word):
    return word["feats"] and "Person=3" in word["feats"] \
        and "Tense=Pres" in word["feats"] \
        and "VerbForm=Fin" in word["feats"] \
        and "Number" in word["feats"]


def __perturb_hop_words(sent, num_hops, marker_sg, marker_pl,lang):
    perturbed_tokens, _ = __perturb_hop_words_complete_hops(
        sent, num_hops, marker_sg, marker_pl,lang)
    return perturbed_tokens


def check_word_hops_completed(sent, lang, num_hops=4, marker=MARKER_HOP_SING):
    _, hops_completed = __perturb_hop_words_complete_hops(
        sent, num_hops, marker, marker,lang)
    return hops_completed


def __perturb_hop_words_complete_hops(sent, num_hops, marker_sg, marker_pl, lang):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()
    tokenizer = TOKENIZATIONER[lang]['hop']
    hop_completed = []
    new_sent = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            new_sent.append(
                word["lemma"] if word["lemma"] is not None else word["text"])

            # Marker hopping logic
            insert_index = len(new_sent)-1
            skipped_words = 0
            while skipped_words < num_hops and insert_index > 0:

                # Handle edge case when punctuation (or sequence of
                # punctuation) begin the sentence
                if (not any([c.isalnum() for c in
                             "".join(new_sent[:insert_index])])):
                    break

                # Count word as skipped if it is not a special token
                if (new_sent[insert_index] not in PART_TOKENS) and \
                        (not set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    skipped_words += 1
                insert_index -= 1

            # Handle edge case when insert index is punctuation (and this is not
            # sentence-initial punctuation)
            if any([c.isalnum() for c in
                    "".join(new_sent[:insert_index])]):
                while insert_index != 0 and (new_sent[insert_index] in PART_TOKENS
                                             or set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    insert_index -= 1

            # Handle edge case when token before insert index is part/aux token
            if insert_index != 0 and new_sent[insert_index-1] in PART_TOKENS:
                insert_index -= 1

            # Log if this sentence had all full hops
            hop_completed.append(skipped_words == num_hops)

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                new_sent.insert(insert_index, marker_sg)
            elif "Number=Plur" in word["feats"]:
                new_sent.insert(insert_index, marker_pl)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

        else:
            new_sent.append(word["text"])

    new_sent.reverse()
    sent_string = " ".join(merge_part_tokens(new_sent))

    tokens = tokenizer.encode(sent_string)
    return tokens, all(hop_completed) and len(hop_completed) > 0


def __perturb_hop_tokens(sent, num_hops, lang):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()
    tokenizer = TOKENIZATIONER[lang]['hop']
    new_sent = deque()
    tokens = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            lemma = word["lemma"] if word["lemma"] is not None else word["text"]

            if len(new_sent) > 0 and new_sent[0] in PART_TOKENS:
                lemma = lemma + new_sent[0]
                new_sent.popleft()

            if len(new_sent) > 0:
                sent_string = " ".join(merge_part_tokens(new_sent))
                tokens = tokenizer.encode(
                    " " + sent_string) + tokens

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                tokens.insert(num_hops, marker_sg_token)
            elif "Number=Plur" in word["feats"]:
                tokens.insert(num_hops, marker_pl_token)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

            new_sent = deque()
            new_sent.append(lemma)

        else:
            new_sent.appendleft(word["text"])

    if len(new_sent) > 0:
        sent_string = " ".join(merge_part_tokens(new_sent))
        tokens = tokenizer.encode(sent_string) + tokens
    return tokens


def __perturb_reverse(sent, rng, reverse, full, lang):
    tokenizer = TOKENIZATIONER[lang]['reverse']
    # Get sentence text and GPT-2 tokens
    tokens = tokenizer.encode(sent["sent_text"])

    # Pick random index to insert REV token
    i = rng.choice(len(tokens)+1)
    tokens.insert(i, marker_rev_token)

    # Extract tokens before/after the marker, and reverse tokens after
    tokens_before = tokens[:i+1]
    tokens_after = tokens[i+1:]
    if reverse:
        tokens_after.reverse()
    new_tokens = tokens_before + tokens_after
    if full:
        assert not reverse
        new_tokens.reverse()

    return new_tokens


def __perturb_shuffle_deterministic(sent, seed, shuffle, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    if lang=='ZH':
        sent_text = ''.join(sent["sent_text"].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])
    if shuffle:
        default_rng(seed).shuffle(tokens)
    return tokens

def __perturb_shuffle_deterministic_word(sent, seed, shuffle, lang):
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    tokens = sent["sent_text"].split()
    if shuffle:
        default_rng(seed).shuffle(tokens)
        tokens = tokenizer.encode(' '.join(tokens))
    return tokens


def __perturb_shuffle_nondeterministic(sent, rng, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    rng.shuffle(tokens)
    return tokens


def __perturb_shuffle_local(sent, seed, lang, window=5):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    if lang =='ZH':
        sent_text = ''.join(sent['sent_text'].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])

    # Shuffle tokens in batches of size window
    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()
        default_rng(seed).shuffle(batch)
        shuffled_tokens += batch

    return shuffled_tokens


def __perturb_shuffle_local_word(sent, seed, lang, window=5):
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    tokens = sent['sent_text'].split()

    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()  
        default_rng(seed).shuffle(batch) 
        shuffled_tokens += batch
    shuffled_words = tokenizer.encode(' '.join(shuffled_tokens))
    return shuffled_words

def __perturb_shuffle_even_odd(sent, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
    odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
    return even + odd


##############################################################################
# AFFECT FUNCTIONS
# These functions define when a perturbation has been applied to a sentence
# not. This is used for identifying which test sentences have been
# altered to separate affected vs. unaffected senences. Affect functions are
# functions of the input sentence object and return a boolean.
##############################################################################


def affect_hop(sent,lang):
    return any([__affect_hop_word(word) for word in sent['word_annotations']]) \
        and sent["constituency_parse"] is not None


def affect_reverse(sent,lang):
    return True


def affect_shuffle(sent,lang):
    return True


def affect_none(sent,lang):
    return False


##############################################################################
# FILTER FUNCTIONS
# These functions define when an affected sentence should be included in the
# final dataset. For instance, hop perturbations where the marker is placed
# at the end of the sentence should be excluded. A filter function returns
# True if an affected sentence should be included in the dataset.
##############################################################################


def filter_hop(sent,lang):
    # Assertion needed since filter function is only defined for affected
    # sentences
    assert (affect_hop(sent,lang))
    return check_word_hops_completed(sent, 4)


def filter_reverse(sent,lang):
    return True


def filter_shuffle(sent, lang):
    tokenizer = TOKENIZATIONER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    return len(tokens) > 1 and len(tokens) <= 350


def filter_none(sent,lang):
    return False


##############################################################################
# PERTURBATION FUNCTIONS
# These functions define how a perturbation will affect a sentence. They
# take in a sentence object and an optional marker
# for verb transformations. They return a string representing the transformed
# sentence.
##############################################################################


def perturb_hop_words4(sent, lang):
    return __perturb_hop_words(sent, 4, MARKER_HOP_SING, MARKER_HOP_PLUR, lang)


def perturb_hop_tokens4(sent,lang):
    return __perturb_hop_tokens(sent, 4,lang)


def perturb_hop_control(sent,lang):
    return __perturb_hop_tokens(sent, 0,lang)


def perturb_reverse(sent, rng, lang, reverse=True, full=False):
    return __perturb_reverse(sent, rng, reverse, full,lang)


def perturb_shuffle_deterministic(sent, lang, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic(sent, seed, shuffle,lang)

def perturb_shuffle_deterministic_word(sent, lang, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic_word(sent, seed, shuffle,lang)

def perturb_shuffle_nondeterministic(sent, rng, lang):
    return __perturb_shuffle_nondeterministic(sent, rng, lang)


def perturb_shuffle_local(sent, seed, window, lang):
    return __perturb_shuffle_local(sent, seed, lang, window)

def perturb_shuffle_local_word(sent, seed, window, lang):
    return __perturb_shuffle_local_word(sent, seed, lang, window)


def perturb_shuffle_even_odd(sent, lang):
    return __perturb_shuffle_even_odd(sent, lang)

TOKENIZER_DICT = {
       "EN": "gpt2",
       "DE": "dbmdz/german-gpt2",
       "RU": "sberbank-ai/rugpt3large_based_on_gpt2",
       "RO": "dumitrescustefan/gpt-neo-romanian-780m",
       "TR": "ytu-ce-cosmos/turkish-gpt2",
       "FR": "Cedille/fr-boris",
       "NL": "yhavinga/gpt-neo-125M-dutch",
       "IT": "iGeniusAI/Italia-9B-Instruct-v0.1",
       "PL":"flax-community/papuGaPT2",
       "ZH": "hfl/chinese-bert-wwm",
        }

def test_tokenizer(tokenizer):
    print(tokenizer)
    print('a')
    print(len(tokenizer))
    print(tokenizer.encode('<|endoftext|>'))
gpt2_tokenizer_de = get_gpt2_tokenizer_with_markers([], 'DE') 
gpt2_tokenizer_en = get_gpt2_tokenizer_with_markers([],'EN')
gpt2_tokenizer_tr = get_gpt2_tokenizer_with_markers([],'TR')
gpt2_tokenizer_ru = get_gpt2_tokenizer_with_markers([], 'RU')
gpt2_tokenizer_ro = get_gpt2_tokenizer_with_markers([], 'RO')
gpt2_tokenizer_fr = get_gpt2_tokenizer_with_markers([], 'FR') 
gpt2_tokenizer_nl = get_gpt2_tokenizer_with_markers([], 'NL')
gpt2_tokenizer_pl = get_gpt2_tokenizer_with_markers([], 'PL')
gpt2_tokenizer_it = get_gpt2_tokenizer_with_markers([], 'IT')
gpt2_tokenizer_zh = get_gpt2_tokenizer_with_markers([], 'ZH')
test_tokenizer(gpt2_tokenizer_zh)
gpt2_hop_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [MARKER_HOP_SING, MARKER_HOP_PLUR], 'EN')
gpt2_rev_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [MARKER_REV], 'EN')
# Get ids of marker tokens
marker_rev_token = gpt2_rev_tokenizer_en.get_added_vocab()[
           MARKER_REV]
gpt2_det_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [BOS_TOKEN], 'EN')
#GPT-2 reverse tokenization
#gpt2_rev_tokenizer_es = get_gpt2_tokenizer_with_markers([MARKER_REV], 'ES')  
# Get ids of marker tokens
bos_token_id = gpt2_det_tokenizer_en.get_added_vocab()[BOS_TOKEN]
marker_sg_token = gpt2_hop_tokenizer_en.get_added_vocab()[
           MARKER_HOP_SING]
marker_pl_token = gpt2_hop_tokenizer_en.get_added_vocab()[
           MARKER_HOP_PLUR]
# GPT-2 determiner tokenization
# Get id of BOS token
#gpt2_original_tokenizer = get_gpt2_tokenizer_with_markers([],)
##############################################################################
# PERTURBATIONS
# This dict maps the name of a perturbation to its perturbation and filter
# functions. The names and functions in this dict are used throughout the
# repo.
##############################################################################


TOKENIZATIONER = {
'EN':{"shuffle": gpt2_tokenizer_en},
"DE":{"shuffle": gpt2_tokenizer_de},
"RU":{"shuffle": gpt2_tokenizer_ru},
"TR":{"shuffle": gpt2_tokenizer_tr},
"RO":{"shuffle": gpt2_tokenizer_ro},
"FR":{"shuffle": gpt2_tokenizer_fr},
"NL":{"shuffle": gpt2_tokenizer_nl},
"IT":{"shuffle": gpt2_tokenizer_it},
"ZH":{"shuffle": gpt2_tokenizer_zh},
"PL":{"shuffle": gpt2_tokenizer_pl},
}
PERTURBATIONS = {
    "shuffle_control_en": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='EN', seed=None, shuffle=False),
        "lang": 'en',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#606060",
    },
    "shuffle_control_fr": {
    "perturbation_function": partial(perturb_shuffle_deterministic, lang='FR', seed=None, shuffle=False),
    "lang": 'fr',
    "affect_function": affect_shuffle,
    "filter_function": filter_shuffle,
    "gpt2_tokenizer": TOKENIZATIONER['FR']['shuffle'],
    "color": "#606060",
                                                            },
"shuffle_control_de": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='DE', seed=None, shuffle=False),
        "lang": 'de',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['DE']['shuffle'],
        "color": "#606060",
    },
"shuffle_control_nl": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='NL', seed=None, shuffle=False),
        "lang": 'nl',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['NL']['shuffle'],
        "color": "#606060",
        },
"shuffle_control_it": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='IT', seed=None, shuffle=False),
         "lang": 'it',
         "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['IT']['shuffle'],
         "color": "#606060",
                },
"shuffle_control_tr": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='TR', seed=None, shuffle=False),
        "lang": 'tr',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
        "color": "#606060",
    },
"shuffle_control_zh": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='ZH', seed=None, shuffle=False),
        "lang": 'zh',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['ZH']['shuffle'],
        "color": "#606060",
    },
"shuffle_control_ro": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='RO', seed=None, shuffle=False),
        "lang": 'ro',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['RO']['shuffle'],
        "color": "#606060",
    },
"shuffle_control_ru": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='RU', seed=None, shuffle=False),
        "lang": 'ru',
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['RU']['shuffle'],
        "color": "#606060",
    },
    "shuffle_local3_en": {
        "perturbation_function": partial(perturb_shuffle_local, lang='EN', seed=0, window=3),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#208EA3",
    },
    "shuffle_nondeterministic_en": {
        "perturbation_function": partial(perturb_shuffle_nondeterministic, lang='EN', rng=default_rng(0)),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#E8384F",
    },
  "shuffle_nondeterministic_tr": {
        "perturbation_function": partial(perturb_shuffle_nondeterministic, lang='TR', rng=default_rng(0)),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
        "color": "#E8384F",
    },
 "shuffle_nondeterministic_tr": {
        "perturbation_function": partial(perturb_shuffle_nondeterministic, lang='TR', rng=default_rng(0)),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
            "color": "#E8384F",
                },
    "shuffle_deterministic21_en": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='EN', seed=21, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#FFB000",
    },
        "shuffle_deterministic21_tr": {
            "perturbation_function": partial(perturb_shuffle_deterministic, lang='TR', seed=21, shuffle=True),
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
             "color": "#FFB000",
    },

    "shuffle_deterministic57_en": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='EN',seed=57, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['EN']['shuffle'],
        "color": "#8db000",
    },
    "shuffle_deterministic84_en": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='EN',seed=84, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#62BB35",
    },

    "shuffle_deterministic57_tr": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='TR',seed=57, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['TR']['shuffle'],
        "color": "#8db000",
                                                          },

      "shuffle_deterministic21_word_tr": {
        "perturbation_function": partial(perturb_shuffle_deterministic_word, lang='TR',seed=21, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['TR']['shuffle'],
        "color": "#8db000",
                                                          },
     "shuffle_deterministic84_tr": {
        "perturbation_function": partial(perturb_shuffle_deterministic, lang='TR',seed=84, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
            "color": "#62BB35",
                                                              },
    "shuffle_local5_en": {
        "perturbation_function": partial(perturb_shuffle_local, lang='EN',seed=0, window=5),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['EN']['shuffle'],
        "color": "#4178BC",
    },
    "shuffle_local10_en": {
        "perturbation_function": partial(perturb_shuffle_local, lang='EN',seed=0, window=10),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#AA71FF",
    },
    "shuffle_even_odd_en": {
        "perturbation_function": partial(perturb_shuffle_even_odd, lang='EN'),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['EN']['shuffle'],
        "color": "#E37CFF",
    },

    "shuffle_local5_tr": {
        "perturbation_function": partial(perturb_shuffle_local, lang='TR',seed=0, window=5),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['TR']['shuffle'],
         "color": "#4178BC",
                                                            },
    "shuffle_local10_tr": {
        "perturbation_function": partial(perturb_shuffle_local, lang='TR',seed=0, window=10),
         "affect_function": affect_shuffle,
         "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
         "color": "#AA71FF",
                             },
        "shuffle_even_odd_tr": {
        "perturbation_function": partial(perturb_shuffle_even_odd, lang='TR'),
        "affect_function": affect_shuffle,
         "filter_function": filter_shuffle,
        "gpt2_tokenizer": TOKENIZATIONER['TR']['shuffle'],
        "color": "#E37CFF",},
        "shuffle_local3_tr": {
        "perturbation_function": partial(perturb_shuffle_local, lang='TR',seed=0, window=3),
         "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
          "gpt2_tokenizer":TOKENIZATIONER['TR']['shuffle'],
         "color": "#4178BC",},

        "shuffle_local3_word_tr": {
         "perturbation_function": partial(perturb_shuffle_local_word, lang='TR',seed=0, window=3),
         "affect_function": affect_shuffle,
         "filter_function": filter_shuffle,
        "gpt2_tokenizer":TOKENIZATIONER['TR']['shuffle'],
         "color": "#4178BC",},
                                                                                                                     
    # "reverse_control": {
    #     "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=False, full=False),
    #     "affect_function": affect_reverse,
    #     "filter_function": filter_reverse,
    #     "gpt2_tokenizer": gpt2_rev_tokenizer,
    #     "color": "#606060",
    # },
    # "reverse_partial": {
    #     "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=True, full=False),
    #     "affect_function": affect_reverse,
    #     "filter_function": filter_reverse,
    #     "gpt2_tokenizer": gpt2_rev_tokenizer,
    #     "color": "#E5A836",
    # },
    # "reverse_full": {
    #     "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=False, full=True),
    #     "affect_function": affect_reverse,
    #     "filter_function": filter_reverse,
    #     "gpt2_tokenizer": gpt2_rev_tokenizer,
    #     "color": "#A348A6",
    # },
    # "hop_control": {
    #     "perturbation_function": perturb_hop_control,
    #     "affect_function": affect_hop,
    #     "filter_function": filter_hop,
    #     "gpt2_tokenizer": gpt2_hop_tokenizer,
    #     "color": "#606060",
    # },
    # "hop_tokens4": {
    #     "perturbation_function": perturb_hop_tokens4,
    #     "affect_function": affect_hop,
    #     "filter_function": filter_hop,
    #     "gpt2_tokenizer": gpt2_hop_tokenizer,
    #     "color": "#fa8128",
    # },
    # "hop_words4": {
    #     "perturbation_function": perturb_hop_words4,
    #     "affect_function": affect_hop,
    #     "filter_function": filter_hop,
    #     "gpt2_tokenizer": gpt2_hop_tokenizer,
    #     "color": "#03a0ff",
    # },
}
