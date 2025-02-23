from glob import glob
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
perplexity_result = pd.read_csv('perplexity_results_new.csv')
def perp_translate(row):
    if 'shuffle_control' in row['perturb']:
        return 1
    else:
        return 0
SEEDS = [41,53,81]  # Single seed case
CHECKPOINTS = list(range(300, 1200 + 1, 100))

perplexity_result['possible'] = perplexity_result.apply(perp_translate, axis=1)
perplexity_result.drop(perplexity_result[perplexity_result.perturb.str.contains('adj')|perplexity_result.lang.str.contains('RN')].index, inplace=True)
perplexity_results600 = perplexity_result.drop(perplexity_result[perplexity_result['checkpoint800']>600].index)
line_style = {0:[' ', '-', 0.5], 1:['.','-',1.2]}

all_langs_below600 = set([(str(row["lang"]).lower(), row["lang"], row["perturb"], line_style[row['possible']][0], line_style[row['possible']][1],str(row["lang"]).lower()+'-'+row["perturb"], row['perturb'],line_style[row['possible']][2]) for _, row in perplexity_results600.iterrows()])
all_langs = list(set([(str(row["lang"]).lower(), row["lang"], row["perturb"], line_style[row['possible']][0], line_style[row['possible']][1],str(row["lang"]).lower()+'-'+row["perturb"],line_style[row['possible']][2]) for _, row in perplexity_result.iterrows()]))
natural = [x for x in all_langs if 'shuffle_control' in x[2]]
all_langs+=natural
# all_langs=natural
PERTURBATION = {
    'shuffle_control': {
        'ar':'#D63A3A',
        'tr':'#FF9900',
        'ru':'#FFDC00',
        'pl': '#BCBD22',
        'de': '#2CA030',

        'it': '#1C5319',
        'pt': '#12A2A8',
        'nl': '#1F83B4',
        'ro': '#C7519C',


        'en': '#F3B0E8',
        'fr': '#AB88BA',
         'zh': '#4C205F',

    }
}
def plot_mean_perplexities(ax, file_info, title, checkpoints, seeds, PERTURBATION):
    results_path = 'perplexity_results/{}_{}/randinit_seed{}_test_{}_{}_pretrained.csv'

    for file_data in file_info:
        lang, lang2, permutation, marker, linestyle, legend_name, line_width = file_data
        if permutation!='shuffle_control':
            legend_name = permutation
        all_seeds_gmeans = []
        print(lang, permutation)
        for seed in seeds:
            df = pd.read_csv(results_path.format(permutation, lang2, seed, permutation, lang), lineterminator='\n')
            gmeans = [stats.gmean(df[f"Perplexities (ckpt {ckpt})"]) for ckpt in checkpoints]
            all_seeds_gmeans.append(gmeans)
        all_seeds_gmeans = np.array(all_seeds_gmeans)
        means = np.mean(all_seeds_gmeans, axis=0)
        if len(seeds) > 1:
            means = np.mean(all_seeds_gmeans, axis=0)
            print(means)
            sems = stats.sem(all_seeds_gmeans, axis=0)
            ci = 1.96 * sems
            ci_lower = means - ci
            ci_upper = means + ci
            ci = (ci_upper - ci_lower) / 2
        else:
            ci = None
        if permutation =='shuffle_control':
            colorp = PERTURBATION[f'shuffle_control'][lang]
        else:
            colorp = '#C4D8F3'
        if ci is not None:
            ax.errorbar(checkpoints, means, yerr=ci, marker=marker, markersize=4, linewidth=line_width,
                        color=colorp,
                        linestyle=linestyle, label=legend_name)
        else:
            ax.plot(checkpoints, means, marker=marker, markersize=4, linewidth=0.8,
                    color=colorp,
                    linestyle=linestyle, label=legend_name)

        ax.set_title(title)
        ax.grid(True, color="#E5E5E5")
        ax.legend(fontsize=5, framealpha=1)


def plot_perplexities_single(file_infos, title, checkpoints, seeds, color, output_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 3]})

    plot_mean_perplexities(ax1, file_infos, title, checkpoints, seeds, color)
    plot_mean_perplexities(ax2, file_infos, title, checkpoints, seeds, color)

    # Set different y-limits
    ax1.set_ylim(600, 7500)  # Upper part of the y-axis (zoomed out)
    ax2.set_ylim(0, 600)  # Lower part of the y-axis (detailed view)

    # Hide spines between the two subplots
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='y', labelsize=5)  # Adjust y-tick label size for ax1
    ax2.tick_params(axis='y', labelsize=7)  # Adjust y-tick label size for ax2

    d = .005
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.legend().set_visible(False)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax2.legend().set_visible(False)
    # ax2.set_ylabel("Geometric Mean Perplexity")
    ax2.set_xlabel("Training Steps")
    fig.text(0.04, 0.5, "Geometric Mean Perplexity of Test Set", va='center', rotation='vertical', fontsize=10)
    fig.suptitle(title)
    # custom_labels = [lang.upper() for lang in PERTURBATION['shuffle_control'].keys()]+['Impossible']
    custom_labels = ['Arabic', 'Turkish', 'Russian', 'Polish', 'German', 'Italian', 'Portuguese', 'Dutch', 'Romanian', 'English', 'French', 'Chinese', 'Impossible']
    custom_colors = [color for color in PERTURBATION['shuffle_control'].values()]+['#C4D8F3']

    legend_handles = [Line2D([0], [0], color=color, lw=2) for color in custom_colors]
    fig.legend(legend_handles, custom_labels, title="Experiments", loc='center left', bbox_to_anchor=(0.93, 0.5),
               fontsize=8)

    plt.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Plot saved as {output_name}.pdf")


title =''


plot_perplexities_single(all_langs, title, CHECKPOINTS, SEEDS, PERTURBATION,'more_specific.pdf')
