import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from matplotlib import cm

# A gradient of 5 shades of blue
SEEDS = [41, 53, 81]  # Single seed case
CHECKPOINTS = list(range(300, 1200 + 1, 100))

titles = ["English", "German", "Turkish", "Italian", "Dutch", "Chinese", "Arabic", "Polish", "Romanian", "Russian",
          "French", 'Portuguese']
ling_info = [
    # English (EN)
    [
        ("en", "EN", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("en", "EN", "shuffle_deterministic84", "v", "-", 'shuffle_deterministic84'),
        ("en", "EN", "shuffle_deterministic57", "v", ":", 'shuffle_deterministic57'),
        ("en", "EN", "shuffle_deterministic21", "v", "--", 'shuffle_deterministic21'),
        ("en", "EN", "shuffle_local10", "^", "-.", 'shuffle_local(w=10)'),
        ("en", "EN", "shuffle_local5", "^", "--", 'shuffle_local(w=5)'),
        ("en", "EN", "shuffle_local3", "^", "-", 'shuffle_local(w=3)'),
        ("en", "EN", "shuffle_local2", "^", "-", 'shuffle_local(w=2)'),
        ("en", "EN", "shuffle_even_odd", "d", "-.", 'shuffle_even_odd'),
        ("en", "EN", "shuffle_control", "o", "-", 'attested')],  # Diamond, Dash-Dot

    [
        ("de", "DE", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("de", "DE", "shuffle_deterministic84", "v", "-", 'de-shuffle_deterministic84'),
        ("de", "DE", "shuffle_deterministic57", "v", ":", 'de-shuffle_deterministic57'),
        ("de", "DE", "shuffle_deterministic21", "v", "--", 'de-shuffle_deterministic21'),
        ("de", "DE", "shuffle_local10", "^", "-.", 'de-shuffle_local10'),
        ("de", "DE", "shuffle_local5", "^", "--", 'de-shuffle_local5'),
        ("de", "DE", "shuffle_local3", "^", "-", 'de-shuffle_local3'),
        ("de", "DE", "shuffle_local2", "^", "-", 'de-shuffle_local2'),
        ("de", "DE", "shuffle_even_odd", "d", "-.", 'de-shuffle_even_odd'),
        ("de", "DE", "shuffle_control", "o", "-", 'de-no_shuffle'), ],

    #
    # Turkish (TR)
    [
        ("tr", "TR", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("tr", "TR", "shuffle_deterministic84", "v", "-", 'tr-shuffle_deterministic84'),
        ("tr", "TR", "shuffle_deterministic57", "v", ":", 'tr-shuffle_deterministic57'),
        ("tr", "TR", "shuffle_deterministic21", "v", "--", 'tr-shuffle_deterministic21'),
        ("tr", "TR", "shuffle_local10", "^", "-", 'tr-shuffle_local10'),
        ("tr", "TR", "shuffle_local5", "^", "-", 'tr-shuffle_local5'),
        ("tr", "TR", "shuffle_local3", "^", "-", 'tr-shuffle_local3'),
        ("tr", "TR", "shuffle_local2", "^", "-", 'tr-shuffle_local2'),
        ("tr", "TR", "shuffle_control", "o", "-", 'tr-no_shuffle'), ],
    # # # Italian (IT)
    [
        ("it", "IT", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("it", "IT", "shuffle_deterministic84", "v", "-", 'it-shuffle_deterministic84'),
        ("it", "IT", "shuffle_deterministic57", "v", ":", 'it-shuffle_deterministic57'),
        ("it", "IT", "shuffle_deterministic21", "v", "--", 'it-shuffle_deterministic21'),
        ("it", "IT", "shuffle_local10", "^", "-.", 'it-shuffle_local10'),
        ("it", "IT", "shuffle_local5", "^", "--", 'it-shuffle_local5'),
        ("it", "IT", "shuffle_local3", "^", "-", 'it-shuffle_local3'),
        ("it", "IT", "shuffle_local2", "^", "-", 'it-shuffle_local2'),
        ("it", "IT", "shuffle_even_odd", "d", "-.", 'it-shuffle_even_odd'),
        ("it", "IT", "shuffle_control", "o", "-", 'it-no_shuffle'), ],

    # # Dutch (NL)
    [
        ("nl", "NL", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("nl", "NL", "shuffle_deterministic84", "v", "-", 'nl-shuffle_deterministic84'),
        ("nl", "NL", "shuffle_deterministic57", "v", ":", 'nl-shuffle_deterministic57'),
        ("nl", "NL", "shuffle_deterministic21", "v", "--", 'nl-shuffle_deterministic21'),
        ("nl", "NL", "shuffle_local10", "^", "-.", 'nl-shuffle_local10'),
        ("nl", "NL", "shuffle_local5", "^", "--", 'nl-shuffle_local5'),
        ("nl", "NL", "shuffle_local3", "^", "-", 'nl-shuffle_local3'),
        ("nl", "NL", "shuffle_local2", "^", "-", 'nl-shuffle_local2'),
        ("nl", "NL", "shuffle_even_odd", "d", "-.", 'nl-shuffle_even_odd'),
        ("nl", "NL", "shuffle_control", "o", "-", 'nl-no_shuffle'), ],

    # # # # Chinese (ZH)
    [
        ("zh", "ZH", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("zh", "ZH", "shuffle_deterministic84", "v", "-", 'zh-shuffle_deterministic84'),
        ("zh", "ZH", "shuffle_deterministic57", "v", ":", 'zh-shuffle_deterministic57'),
        ("zh", "ZH", "shuffle_deterministic21", "v", "--", 'zh-shuffle_deterministic21'),
        ("zh", "ZH", "shuffle_local10", "^", "-.", 'zh-shuffle_local10'),
        ("zh", "ZH", "shuffle_local5", "^", "--", 'zh-shuffle_local5'),
        ("zh", "ZH", "shuffle_local3", "^", "-", 'zh-shuffle_local3'),
        ("zh", "ZH", "shuffle_local2", "^", "-", 'zh-shuffle_local2'),
        ("zh", "ZH", "shuffle_even_odd", "d", "-.", 'zh-shuffle_even_odd'),
        ("zh", "ZH", "shuffle_control", "o", "-", 'zh-no_shuffle'), ],

    [
        ("ar", "AR", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("ar", "AR", "shuffle_deterministic84", "v", "-", 'ar-shuffle_deterministic84'),
        ("ar", "AR", "shuffle_deterministic57", "v", ":", 'ar-shuffle_deterministic57'),
        ("ar", "AR", "shuffle_deterministic21", "v", "--", 'ar-shuffle_deterministic21'),
        ("ar", "AR", "shuffle_local10", "^", "-.", 'ar-shuffle_local10'),
        ("ar", "AR", "shuffle_local5", "^", "--", 'ar-shuffle_local5'),
        ("ar", "AR", "shuffle_local3", "^", "-", 'ar-shuffle_local3'),
        ("ar", "AR", "shuffle_local2", "^", "-", 'ar-shuffle_local2'),
        ("ar", "AR", "shuffle_even_odd", "d", "-.", 'ar-shuffle_even_odd'),
        ("ar", "AR", "shuffle_control", "o", "-", 'ar-no_shuffle'), ],

    [
        ("pl", "PL", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("pl", "PL", "shuffle_deterministic84", "v", "-", 'pl-shuffle_deterministic84'),
        ("pl", "PL", "shuffle_deterministic57", "v", ":", 'pl-shuffle_deterministic57'),
        ("pl", "PL", "shuffle_deterministic21", "v", "--", 'pl-shuffle_deterministic21'),
        ("pl", "PL", "shuffle_local10", "^", "-.", 'pl-shuffle_local10'),
        ("pl", "PL", "shuffle_local5", "^", "--", 'pl-shuffle_local5'),
        ("pl", "PL", "shuffle_local3", "^", "-", 'pl-shuffle_local3'),
        ("pl", "PL", "shuffle_local2", "^", "-", 'pl-shuffle_local2'),
        ("pl", "PL", "shuffle_even_odd", "d", "-.", 'pl-shuffle_even_odd'),
        ("pl", "PL", "shuffle_control", "o", "-", 'pl-no_shuffle')],

    # # # # Romanian (RO)
    [
        ("ro", "RO", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("ro", "RO", "shuffle_deterministic84", "v", "-", 'ro-shuffle_deterministic84'),
        ("ro", "RO", "shuffle_deterministic57", "v", ":", 'ro-shuffle_deterministic57'),
        ("ro", "RO", "shuffle_deterministic21", "v", "--", 'ro-shuffle_deterministic21'),
        ("ro", "RO", "shuffle_local10", "^", "-.", 'ro-shuffle_local10'),
        ("ro", "RO", "shuffle_local5", "^", "--", 'ro-shuffle_local5'),
        ("ro", "RO", "shuffle_local3", "^", "-", 'ro-shuffle_local3'),
        ("ro", "RO", "shuffle_local2", "^", "-", 'ro-shuffle_local2'),
        ("ro", "RO", "shuffle_even_odd", "d", "-.", 'ro-shuffle_even_odd'),
        ("ro", "RO", "shuffle_control", "o", "-", 'ro-no_shuffle'), ],

    # #
    # # # Russian (RU)
    [
        ("ru", "RU", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("ru", "RU", "shuffle_deterministic84", "v", "-", 'ru-shuffle_deterministic84'),
        ("ru", "RU", "shuffle_deterministic57", "v", ":", 'ru-shuffle_deterministic57'),
        ("ru", "RU", "shuffle_deterministic21", "v", "--", 'ru-shuffle_deterministic21'),
        ("ru", "RU", "shuffle_local10", "^", "-.", 'ru-shuffle_local10'),
        ("ru", "RU", "shuffle_local5", "^", "--", 'ru-shuffle_local5'),
        ("ru", "RU", "shuffle_local3", "^", "-", 'ru-shuffle_local3'),
        ("ru", "RU", "shuffle_local2", "^", "-", 'ru-shuffle_local2'),
        ("ru", "RU", "shuffle_even_odd", "d", "-.", 'ru-shuffle_even_odd'),
        ("ru", "RU", "shuffle_control", "o", "-", 'ru-no_shuffle'), ],

    [
        ("fr", "FR", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("fr", "FR", "shuffle_deterministic84", "v", "-", 'fr-shuffle_deterministic84'),
        ("fr", "FR", "shuffle_deterministic57", "v", ":", 'fr-shuffle_deterministic57'),
        ("fr", "FR", "shuffle_deterministic21", "v", "--", 'fr-shuffle_deterministic21'),

        ("fr", "FR", "shuffle_local10", "^", "-.", 'fr-shuffle_local10'),
        ("fr", "FR", "shuffle_local5", "^", "--", 'fr-shuffle_local5'),
        ("fr", "FR", "shuffle_local3", "^", "-", 'fr-shuffle_local3'),
        ("fr", "FR", "shuffle_local2", "^", "--", 'fr-shuffle_local2'),
        ("fr", "FR", "shuffle_even_odd", "d", "-.", 'fr-shuffle_even_odd'),
        ("fr", "FR", "shuffle_control", "o", "-", 'fr-no_shuffle'), ],
    [
        ("pt", "PT", "perturb_reverse_full_word", "*", "-", 'perturb_reverse_full_word'),
        ("pt", "PT", "shuffle_deterministic84", "v", "-", 'pt-shuffle_deterministic84'),
        ("pt", "PT", "shuffle_deterministic57", "v", ":", 'pt-shuffle_deterministic57'),
        ("pt", "PT", "shuffle_deterministic21", "v", "--", 'pt-shuffle_deterministic21'),
        ("pt", "PT", "shuffle_local10", "^", "-.", 'pt-shuffle_local10'),
        ("pt", "PT", "shuffle_local5", "^", "--", 'pt-shuffle_local5'),
        ("pt", "PT", "shuffle_local3", "^", "-", 'pt-shuffle_local3'),
        ("pt", "PT", "shuffle_local2", "^", "--", 'pt-shuffle_local2'),
        ("pt", "PT", "shuffle_even_odd", "d", "-.", 'pt-shuffle_even_odd'),
        ("pt", "PT", "shuffle_control", "o", "-", 'pt-no_shuffle'), ],
]

PERTURBATIONS = {
    "perturb_reverse_full_word": "#6F63BB",
    "shuffle_deterministic84": "#8A60B0",
    "shuffle_deterministic57": "#C7519C",
    "shuffle_deterministic21": "#D63A3A",
    "shuffle_local10": "#FFBF50",
    "shuffle_local5": "#BCBD22",
    "shuffle_local3": "#78A641",
    "shuffle_local2": "#2CA030",
    "shuffle_even_odd": "#12A2A8",
    "shuffle_control": "#1F83B4"
}


def plot_mean_perplexities_multilingual(ax, file_info, title, checkpoints, seeds, PERTURBATION):
    results_path = 'perplexity_results/{}_{}/randinit_seed{}_test_{}_{}_pretrained.csv'

    for file_data in file_info:
        lang, lang2, permutation, marker, linestyle, legend_name = file_data
        all_seeds_gmeans = []
        print(file_data)

        for seed in seeds:
            df = pd.read_csv(results_path.format(permutation, lang2, seed, permutation, lang), lineterminator='\n')
            gmeans = [stats.gmean(df[f"Perplexities (ckpt {ckpt})"]) for ckpt in checkpoints]
            # for k, ckpt in enumerate(checkpoints):
            # print(legend_name, ckpt, gmeans[k])
            all_seeds_gmeans.append(gmeans)
        all_seeds_gmeans = np.array(all_seeds_gmeans)
        means = np.mean(all_seeds_gmeans, axis=0)
        if len(seeds) > 1:
            means = np.mean(all_seeds_gmeans, axis=0)
            sems = stats.sem(all_seeds_gmeans, axis=0)
            ci = 1.96 * sems
            ci_lower = means - ci
            ci_upper = means + ci
            ci = (ci_upper - ci_lower) / 2
        else:
            ci = None
        colorp = PERTURBATION[f'{permutation}']
        if ci is not None:
            ax.errorbar(checkpoints, means, yerr=ci, marker=marker, markersize=4, linewidth=0.8,
                        color=colorp,
                        linestyle=linestyle, label=legend_name)
        else:
            ax.plot(checkpoints, means, marker=marker, markersize=4, linewidth=0.8,
                    color=colorp,
                    linestyle=linestyle, label=legend_name)

        ax.set_title(title)
        ax.grid(True, color="lightgray")
        ax.legend(fontsize=5, framealpha=1)


def plot_perplexities_multilingual(file_infos, titles, checkpoints, seeds, PERTURBATIONS_m, output_name):
    # Create the figure and a single axis (axs)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.15)
    fig.supylabel('Geometric Mean Perplexity of Test Set', fontsize=18)
    fig.supxlabel("Training Steps", fontsize=18)
    axes = axes.flatten()
    # axes[11].remove()
    for i, (file_info, title) in enumerate(zip(file_infos, titles)):
        plot_mean_perplexities_multilingual(axes[i], file_info, title, checkpoints, seeds, PERTURBATIONS_m)
        axes[i].tick_params(axis='y', labelsize=6)
        axes[i].tick_params(axis='x', labelsize=6)
        axes[i].yaxis.get_offset_text().set_fontsize(2)  # Adjust the offset text font size
        axes[i].legend().set_visible(False)
        # axes[i].legend(title="Experiments", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6, ncol=2, borderaxespad=0.)
    handles, labels = axes[0].get_legend_handles_labels()  # Get legend info from the first plot
    fig.legend(handles, labels, title="Languages", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5,
               fontsize=8)
    plt.savefig(output_name, format="pdf", bbox_inches="tight")

    print(f"Plot saved as {output_name}.pdf")

    plt.show()


plot_perplexities_multilingual(ling_info, titles, CHECKPOINTS, SEEDS, PERTURBATIONS,
                               'single_LM_results.pdf')