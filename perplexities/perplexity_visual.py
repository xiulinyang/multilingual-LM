import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm

# A gradient of 5 shades of blue
SEEDS = [41, 53, 81]  # Single seed case
CHECKPOINTS = list(range(300, 1200 + 1, 100))

'''
# Turkish (ro) - Deterministic Shuffles
("en", "EN", "perturb_num_adj", "^", ":", 'en-perturb_num_adj'),
("en", "EN", "perturb_adj_num", "^", ":", 'en-perturb_adj_num'),  # Up Triangle, Dotted
("en", "EN", "shuffle_deterministic21_word", "s", ":", 'en-shuffle_determ21_word'),  # Square, Dotted
("en", "EN", "shuffle_local_word3", "^", ":", 'en-shuffle_local3_word'),  # Up Triangle, Dotted
("en", "EN", "shuffle_remove_fw", "^", ":", 'en-shuffle_remove_fw'),  # Up Triangle, Dotted
'''

ling_info = [
    # English (EN)
    # ("en", "EN", "shuffle_deterministic21", "v", "-", 'en-shuffle_deterministic21'),  # Down Triangle, Solid
    # ("en", "EN", "shuffle_deterministic57", "v", ":", 'en-shuffle_deterministic57'),  # Down Triangle, Dotted
    # ("en", "EN", "shuffle_deterministic84", "v", "--", 'en-shuffle_deterministic84'),  # Down Triangle, Dashed
    ("en", "EN", "shuffle_local2", "^", "-", 'en-shuffle_local3'),
    ("en", "EN", "shuffle_local3", "^", "-", 'en-shuffle_local3'),  # Up Triangle, Solid
    ("en", "EN", "shuffle_local5", "^", "--", 'en-shuffle_local5'),  # Up Triangle, Dashed
    ("en", "EN", "shuffle_local10", "^", "-.", 'en-shuffle_local10'),  # Up Triangle, Dash-Dot
    ("en", "EN", "shuffle_even_odd", "d", "-.", 'en-shuffle_even_odd'),  # Diamond, Dash-Dot

    #
    # # # German (DE)
    # ("de", "DE", "shuffle_deterministic21", "v", "-", 'de-shuffle_deterministic21'),
    # ("de", "DE", "shuffle_deterministic57", "v", ":", 'de-shuffle_deterministic57'),
    # ("de", "DE", "shuffle_deterministic84", "v", "--", 'de-shuffle_deterministic84'),
    # ("de", "DE", "shuffle_local3", "^", "-", 'de-shuffle_local3'),
    # ("de", "DE", "shuffle_local5", "^", "--", 'de-shuffle_local5'),
    # ("de", "DE", "shuffle_local10", "^", "-.", 'de-shuffle_local10'),
    # ("de", "DE", "shuffle_even_odd", "d", "-.", 'de-shuffle_even_odd'),

    #
    # # Turkish (TR)
    # ("tr", "TR", "shuffle_deterministic21", "v", "-", 'tr-shuffle_deterministic21'),
    # ("tr", "TR", "shuffle_deterministic57", "v", ":", 'tr-shuffle_deterministic57'),
    # ("tr", "TR", "shuffle_deterministic84", "v", "--", 'tr-shuffle_deterministic84'),
    # ("tr", "TR", "shuffle_local3", "^", "-", 'tr-shuffle_local3'),
    # ("tr", "TR", "shuffle_local5", "^", "--", 'tr-shuffle_local5'),
    # ("tr", "TR", "shuffle_local10", "^", "-.", 'tr-shuffle_local10'),
    # ("tr", "TR", "shuffle_even_odd", "d", "-.", 'tr-shuffle_even_odd'),

    # #
    # # # Italian (IT)
    # ("it", "IT", "shuffle_deterministic21", "v", "-", 'it-shuffle_deterministic21'),
    # ("it", "IT", "shuffle_deterministic57", "v", ":", 'it-shuffle_deterministic57'),
    # ("it", "IT", "shuffle_deterministic84", "v", "--", 'it-shuffle_deterministic84'),
    # ("it", "IT", "shuffle_local3", "^", "-", 'it-shuffle_local3'),
    # ("it", "IT", "shuffle_local5", "^", "--", 'it-shuffle_local5'),
    # ("it", "IT", "shuffle_local10", "^", "-.", 'it-shuffle_local10'),
    # ("it", "IT", "shuffle_even_odd", "d", "-.", 'it-shuffle_even_odd'),


    # # Dutch (NL)
    # ("nl", "NL", "shuffle_deterministic21", "v", "-", 'nl-shuffle_deterministic21'),
    # ("nl", "NL", "shuffle_deterministic57", "v", ":", 'nl-shuffle_deterministic57'),
    # ("nl", "NL", "shuffle_deterministic84", "v", "--", 'nl-shuffle_deterministic84'),
    # ("nl", "NL", "shuffle_local3", "^", "-", 'nl-shuffle_local3'),
    # ("nl", "NL", "shuffle_local5", "^", "--", 'nl-shuffle_local5'),
    # ("nl", "NL", "shuffle_local10", "^", "-.", 'nl-shuffle_local10'),
    # ("nl", "NL", "shuffle_even_odd", "d", "-.", 'nl-shuffle_even_odd'),


    # # # # Chinese (ZH)
    # ("zh", "ZH", "shuffle_deterministic21", "v", "-", 'zh-shuffle_deterministic21'),
    # ("zh", "ZH", "shuffle_deterministic57", "v", ":", 'zh-shuffle_deterministic57'),
    # ("zh", "ZH", "shuffle_deterministic84", "v", "--", 'zh-shuffle_deterministic84'),
    # ("zh", "ZH", "shuffle_local3", "^", "-", 'zh-shuffle_local3'),
    # ("zh", "ZH", "shuffle_local5", "^", "--", 'zh-shuffle_local5'),
    # ("zh", "ZH", "shuffle_local10", "^", "-.", 'zh-shuffle_local10'),
    # ("zh", "ZH", "shuffle_even_odd", "d", "-.", 'zh-shuffle_even_odd'),
    # #
    # # #
    # # # # Romanian (RO)
    # ("ro", "RO", "shuffle_deterministic21", "v", "-", 'ro-shuffle_deterministic21'),
    # ("ro", "RO", "shuffle_deterministic57", "v", ":", 'ro-shuffle_deterministic57'),
    # ("ro", "RO", "shuffle_deterministic84", "v", "--", 'ro-shuffle_deterministic84'),
    # ("ro", "RO", "shuffle_local3", "^", "-", 'ro-shuffle_local3'),
    # ("ro", "RO", "shuffle_local5", "^", "--", 'ro-shuffle_local5'),
    # ("ro", "RO", "shuffle_local10", "^", "-.", 'ro-shuffle_local10'),
    # ("ro", "RO", "shuffle_even_odd", "d", "-.", 'ro-shuffle_even_odd'),

    # #
    # # # Russian (RU)
    # # ("ru", "RU", "shuffle_deterministic21", "v", "-", 'ru-shuffle_deterministic21'),
    # ("ru", "RU", "shuffle_deterministic57", "v", ":", 'ru-shuffle_deterministic57'),
    # ("ru", "RU", "shuffle_deterministic84", "v", "--", 'ru-shuffle_deterministic84'),
    # ("ru", "RU", "shuffle_local3", "^", "-", 'ru-shuffle_local3'),
    # ("ru", "RU", "shuffle_local5", "^", "--", 'ru-shuffle_local5'),
    # ("ru", "RU", "shuffle_local10", "^", "-.", 'ru-shuffle_local10'),
    # ("ru", "RU", "shuffle_even_odd", "d", "-.", 'ru-shuffle_even_odd'),

("en", "EN", "shuffle_control", "o", "-", 'en-no_shuffle'),
# ("de", "DE", "shuffle_control", "o", "-", 'de-no_shuffle'),
# ("tr", "TR", "shuffle_control", "o", "-", 'tr-no_shuffle'),
# ("it", "IT", "shuffle_control", "o", "-", 'it-no_shuffle'),



# ("nl", "NL", "shuffle_control", "o", "-", 'nl-no_shuffle'),
# ("zh", "ZH", "shuffle_control", "o", "-", 'zh-no_shuffle'),
# ("ro", "RO", "shuffle_control", "o", "-", 'ro-no_shuffle'),
# ("ru", "RU", "shuffle_control", "o", "-", 'ru-no_shuffle'),
]


PERTURBATIONS = {
    # Red Tones
    "perturb_adj_num":{"color":"#D6DFEC" },
    "perturb_num_adj": {"color": "#D6DFEC"},
    "shuffle_deterministic21": {"color": "#D6DFEC"},  # Light Red
    "shuffle_deterministic57": {"color": "#D6DFEC"},  # Soft Salmon Pink
    "shuffle_remove_fw": {"color": "#D6DFEC"},  # Soft Salmon Pink
    "shuffle_deterministic84": {"color": "#D6DFEC"},  # Light Apricot Orange
    "shuffle_local2":{"color":"#EABF23" },
    "shuffle_local3": {"color": "#D6DFEC"},  # Soft Peach
    "shuffle_local5": {"color": "#D6DFEC"},  # Soft Pastel Yellow
    "shuffle_local10": {"color": "#D6DFEC"},  # Soft Pastel Green
    "shuffle_even_odd": {"color": "#D6DFEC"},  # Soft Light Blue

    "shuffle_deterministic21_word": {"color": "#D6DFEC"},
    "shuffle_local_word3": {"color": "#D6DFEC"},
     "shuffle_control": {
         "EN": "#3B99B1",
         "DE": "#3EAAA6",
         "TR": "#A2C194",
         "IT": "#D9CB80",
         "NL": "#EABF23",
         "ZH": "#E89E16",
         "RO": "#E87900",
         "RU": "#F13D09", },

}


def plot_mean_perplexities_multilingual(ax_i, ax, file_info, title, checkpoints, seeds, PERTURBATION):
    results_path = 'perplexity_results/{}_{}/randinit_seed{}_test_{}_{}_pretrained.csv'

    for file_data in file_info:
        lang, lang2, permutation, marker, linestyle, legend_name = file_data
        all_seeds_gmeans = []

        for seed in seeds:
            df = pd.read_csv(results_path.format(permutation, lang2, seed, permutation, lang), lineterminator='\n')
            # print(f"Data for seed {seed} and {lang}:\n", df.head())  # Debugging line

            gmeans = [stats.gmean(df[f"Perplexities (ckpt {ckpt})"]) for ckpt in checkpoints]
            for k, ckpt in enumerate(checkpoints):
                print(legend_name, ckpt, gmeans[k])

            all_seeds_gmeans.append(gmeans)

        all_seeds_gmeans = np.array(all_seeds_gmeans)
        means = np.mean(all_seeds_gmeans, axis=0)

        # print(f"Means for {lang}: {means}")  # Debugging line

        # If more than one seed, calculate confidence intervals
        if len(seeds) > 1:
            ci = stats.sem(all_seeds_gmeans, axis=0)
        else:
            ci = None  # No confidence interval for a single seed

        if permutation =='shuffle_control':
            colorp = PERTURBATION[f'{permutation}'][lang2]
        else:
            colorp = PERTURBATION[f'{permutation}']["color"]
        if ci is not None:
            ax.errorbar(checkpoints, means, yerr=ci, marker=marker, markersize=4, linewidth=0.8,
                        color= colorp,
                        linestyle=linestyle, label=legend_name)
        else:
            ax.plot(checkpoints, means, marker=marker, markersize=4, linewidth=0.8,
                    color=colorp,
                    linestyle=linestyle, label=legend_name)

        ax.set_title(title)
        ax.set_ylabel("Geomeroic Mean Perplexity", fontsize=12)
        ax.grid(True, color="lightgray")
        ax.legend(fontsize=5, framealpha=1)


def plot_perplexities_multilingual(file_info, title, checkpoints, seeds, PERTURBATIONS_m, output_name):
    # Create the figure and a single axis (axs)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(wspace=0.15)
    fig.supxlabel("Training Steps", fontsize=12)
    plot_mean_perplexities_multilingual(0, ax, file_info, title, checkpoints, seeds, PERTURBATIONS_m)
    ax.legend(title="Experiments", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6, ncol=2, borderaxespad=0.)
    plt.savefig(output_name, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_name}.pdf")

    plt.show()


plot_perplexities_multilingual(ling_info, "Natural Languages vs Impossible Languages", CHECKPOINTS, SEEDS, PERTURBATIONS,
                               'LM.pdf')