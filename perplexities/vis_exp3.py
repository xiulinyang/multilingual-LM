import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from matplotlib import cm
SEEDS = [41, 53, 81]

CHECKPOINTS = list(range(300, 1200 + 1, 100))
lang_info = [

    [
    ("en", "EN", "shuffle_local3", "^", "-", 'en-shuffle_local3', "shuffle_local (w=3)"),
    ("en", "EN", "shuffle_local2", "^", "-", 'en-shuffle_local2', "shuffle_local (w=2)"),
    ("en", "EN", "shuffle_control", "o", "-", 'en-shuffle_control', 'attested'),
    ("en", "EN", "perturb_reverse_full_word", "*", "-.", 'en-perturb_reverse_full_word', "Reverse_full"),
    ("enrn", "ENRN", "shuffle_control", "o", "-", 'en-np_shuffle_random', 'np-random'),
    ("en", "EN", "perturb_adj_num_np_det", "d", "-.", 'en-perturb_adj_num_np_det', "perturb_annd",),
    ("en", "EN", "perturb_det_num_np_adj", "d", "-.", 'en-perturb_det_num_np_adj', 'perturb_dnna'),
    ("en", "EN", "perturb_np_num_det_adj", "d", "-.", 'en-perturb_np_num_det_adj', 'perturb_nnda'),
    ("en", "EN", "perturb_det_adj_np_num", "d", "-.", 'en-perturb_det_adj_np_num', 'perturb_dann'),
    ("en", "EN", "perturb_det_num_adj_np", "d", "-.", 'en-perturb_det_num_adj_np', 'perturb_dnan'),

    ],

   [
       ("it", "IT", "shuffle_local3", "^", "-", 'it-shuffle_local3', "shuffle_local3"),
    ("it", "IT", "shuffle_local2", "^", "-", 'it-shuffle_local2', "shuffle_local2"),
    ("it", "IT", "shuffle_control", "o", "-", 'it-shuffle_control', 'shuffle_control'),
    ("it", "IT", "perturb_reverse_full_word", "*", "-.", 'it-perturb_reverse_full_word', "perturb_reverse_full_word"),
    ("itrn", "ITRN", "shuffle_control", "o", "-", 'it-np_shuffle_random', 'np-shuffle-random'),
    ("it", "IT", "perturb_adj_num_np_det", "d", "-.", 'it-perturb_adj_num_np_det', "perturb_adj_num_np_det",),
     ("it", "IT", "perturb_det_num_np_adj", "d", "-.", 'it-perturb_det_num_np_adj', 'perturb_det_num_np_adj'),
    ("it", "IT", "perturb_np_num_det_adj", "d", "-.", 'it-perturb_np_num_det_adj', 'perturb_np_num_det_adj'),
    ("it", "IT", "perturb_det_adj_np_num", "d", "-.", 'it-perturb_det_adj_np_num', 'perturb_det_adj_np_num'),
    ("it", "IT", "perturb_det_num_adj_np", "d", "-.", 'it-perturb_det_num_adj_np', 'perturb_det_num_adj_np'),
],

 [
    ("zh", "ZH", "shuffle_local3", "^", "-", 'zh-shuffle_local3', "shuffle_local3"),
    ("zh", "ZH", "shuffle_local2", "^", "-", 'zh-shuffle_local2', "shuffle_local2"),
    ("zh", "ZH", "shuffle_control", "o", "-", 'zh-shuffle_control', 'shuffle_control'),
    ("zh", "ZH", "perturb_reverse_full_word", "*", "-.", 'zh-perturb_reverse_full_word', "perturb_reverse_full_word"),
    ("zhrn", "ZHRN", "shuffle_control", "o", "-", 'zh-np_shuffle_random', 'np-shuffle-random'),
   ("zh", "ZH", "perturb_adj_num_np_det", "d", "-.", 'zh-perturb_adj_num_np_det', "perturb_adj_num_np_det",),
     ("zh", "ZH", "perturb_det_num_np_adj", "d", "-.", 'zh-perturb_det_num_np_adj', 'perturb_det_num_np_adj'),
    ("zh", "ZH", "perturb_np_num_det_adj", "d", "-.", 'zh-perturb_np_num_det_adj', 'perturb_np_num_det_adj'),
  ("zh", "ZH", "perturb_det_num_adj_np", "d", "-.", 'zh-perturb_det_num_adj_np', 'perturb_det_num_adj_np'),
   ("zh", "ZH", "perturb_det_adj_np_num", "d", "-.", 'zh-perturb_det_adj_np_num', 'perturb_det_adj_np_num'),

   ],

[
    ("pt", "PT", "shuffle_local3", "^", "-", 'pt-shuffle_local3', "shuffle_local (w=3)"),
    ("pt", "PT", "shuffle_local2", "^", "-", 'pt-shuffle_local2', "shuffle_local (w=2)"),
    ("pt", "pt", "shuffle_control", "o", "-", 'pt-shuffle_control', 'attested'),
    ("pt", "PT", "perturb_reverse_full_word", "*", "-.", 'pt-perturb_reverse_full_word', "reverse_full"),
    ("ptrn", "PTRN", "shuffle_control", "o", "-", 'pt-np_shuffle_random', 'np-random'),
    ("pt", "PT", "perturb_adj_num_np_det", "d", "-.", 'pt-perturb_adj_num_np_det', "perturb_annd",),
     ("pt", "PT", "perturb_det_num_np_adj", "d", "-.", 'pt-perturb_det_num_np_adj', 'perturb_dnna'),
    ("pt", "PT", "perturb_np_num_det_adj", "d", "-.", 'pt-perturb_np_num_det_adj', 'perturb_nnda'),
    ("pt", "PT", "perturb_det_adj_np_num", "d", "-.", 'pt-perturb_det_adj_np_num', 'perturb_dann'),
    ("pt", "PT", "perturb_det_num_adj_np", "d", "-.", 'pt-perturb_det_num_adj_np', 'perturb_dnan'),

],

]


COLOR = {

    "shuffle_local3": "#52b788",
    "shuffle_local2": "#7b2cbf",
    "perturb_reverse_full_word": "#c77dff",
    "np_shuffle_random": "#a2d6f9",
    "shuffle_control": "#A2B627",
    "perturb_adj_num_np_det": "#ff8800",
    "perturb_det_num_np_adj": "#ffaa00",
    "perturb_np_num_det_adj": "#f1db33",
    "perturb_det_adj_np_num": "#dfab06",
    "perturb_det_num_adj_np": "#da6220",
    "perturb_np_adj_num_det": "#52b788",
}

title = ['English (DNAN)', 'Italian (DNNA)', 'Chinese (DNAN)', 'Portuguese (DNNA)']


def plot_mean_perplexities(ax, file_info, checkpoints, seeds):
    results_path = 'perplexity_results/{}_{}/randinit_seed{}_test_{}_{}_pretrained.csv'
    for file_data in file_info:
        lang, lang2, permutation, marker, linestyle, full_marker, legend_name= file_data
        all_seeds_gmeans = []
        print(file_data)
        for seed in seeds:
            df = pd.read_csv(results_path.format(permutation, lang2, seed, permutation, lang), lineterminator='\n')
            gmeans = [stats.gmean(df[f"Perplexities (ckpt {ckpt})"]) for ckpt in checkpoints]
            for k, ckpt in enumerate(checkpoints):
                print(legend_name, ckpt, gmeans[k])
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
            ci = None  # No confidence interval for a single seed
        color_code = full_marker.split('-')[-1]
        colorp = COLOR[f'{color_code}']
        if ci is not None:
            ax.errorbar(checkpoints, means, yerr=ci, marker=marker, markersize=4, linewidth=0.8,
                        color=colorp,
                        linestyle=linestyle, label=legend_name)
        else:
            ax.plot(checkpoints, means, marker=marker, markersize=4, linewidth=0.8,
                    color=colorp,
                    linestyle=linestyle, label=legend_name)

        ax.grid(True, color="lightgray")
        # ax.legend(fontsize=5, framealpha=1)


def plot_perplexities_grid(file_infos_list, titles, checkpoints, seeds, colors, output_name):
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(20, 5),
                             gridspec_kw={'height_ratios': [1, 3]}, constrained_layout=True)

    for i, (file_infos, title, color) in enumerate(zip(file_infos_list, titles, colors)):
        ax1 = axes[0, i]  # Upper part (zoomed out)
        ax2 = axes[1, i]  # Lower part (detailed view)
        # print(file_infos)
        # Plot on both axes
        plot_mean_perplexities(ax1, file_infos,  checkpoints, seeds)
        plot_mean_perplexities(ax2, file_infos, checkpoints, seeds)

        # Set different y-limits
        if title =='English (DNAN)':
            ax1.set_ylim(150, 500)  # Upper part of the y-axis (zoomed out)
            ax2.set_ylim(50, 150)  # Lower part of the y-axis (detailed view)
        elif title =='Italian (DNNA)':
            ax1.set_ylim(200, 600)  # Upper part of the y-axis (zoomed out)
            ax2.set_ylim(80, 200)  # Lower part of the y-axis (detailed view)
        elif title == 'Chinese (DNAN)':
            ax1.set_ylim(100, 300)  # Upper part of the y-axis (zoomed out)
            ax2.set_ylim(50, 100)  # Lower part of the y-axis (detailed view)
        elif title =='Portuguese (DNNA)':
            ax1.set_ylim(200, 1000)  # Upper part of the y-axis (zoomed out)
            ax2.set_ylim(70, 200)  # Lower part of the y-axis (detailed view)
        ax1.set_title(title, fontsize=18)
        # Hide spines between the two subplots
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')

        # Add diagonal lines to indicate break
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.legend().set_visible(False)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        ax2.legend().set_visible(False)



    # Common xlabel
    fig.supxlabel("Training Steps", size=18)
    fig.supylabel("Geometric Mean Perplexity", size=18)
    # Set the overall figure title
    # fig.suptitle("Comparison of Perplexities Across Languages", fontsize=12)

    # Extract legend handles and labels
    handles, labels = axes[-1][-1].get_legend_handles_labels()
    # Add legend outside the plot
    fig.legend(handles, labels, title="Experiments", loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(labels), fontsize=12)
    # Save & show
    plt.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Plot saved as {output_name}.pdf")


plot_perplexities_grid(lang_info, title, CHECKPOINTS, SEEDS, COLOR, 'typology.pdf')