import numpy as np
import pickle

from blancops.configs.paths import RunPaths
from blancops.math import geometry
from blancops.math import units
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

def plot_train_metrics(results_outdir, dataset):
    run_paths = RunPaths(results_outdir)
    with open(run_paths.train_metrics, 'rb') as f:
        train_metrics = pickle.load(f)
    with open(run_paths.val_metrics, 'rb') as f:
        val_metrics = pickle.load(f)
    
    # Plot Loss, Accuracy, and Angular separation
    nrows = 3 if 'ang_sep' in val_metrics else 2
    fig, axs = plt.subplots(nrows, sharex=True, figsize=(4, 7))
    best_epoch = np.argmin(val_metrics['val_loss'])

    axs[0].plot(train_metrics['epoch'], train_metrics['train_loss'], label='train loss', color='black', linestyle='dotted')
    axs[0].plot(val_metrics['epoch'], val_metrics['val_loss'], label='val loss')
    axs[0].hlines(y=0, xmin=0, xmax=np.max(val_metrics['epoch']), color='red', linestyle='dashed')
    axs[0].vlines(x=best_epoch, ymax=np.max(val_metrics['val_loss']), ymin=0, color='grey', linestyle='dashed', lw=.5)
    axs[0].hlines(y=np.min(val_metrics['val_loss']), xmin=0, xmax=val_metrics['epoch'][-1], linestyle='dashed', color='grey', lw=.5)
    axs[0].set_ylabel('Loss', fontsize=14)
    for i, met_name in enumerate(['candidate_loss', 'filter_loss']):
        if met_name in val_metrics.keys():
            axs[0].plot(train_metrics['epoch'], train_metrics[met_name], color=f'C{i+1}', linestyle='dotted')
            axs[0].plot(val_metrics['epoch'], val_metrics[met_name], color=f'C{i+1}', label='val ' + met_name)
    axs[0].legend(fontsize=10)

    axs[1].plot(train_metrics['epoch'], train_metrics['accuracy'], label='train accuracy', color='black', linestyle='dotted')
    axs[1].plot(val_metrics['epoch'], val_metrics['accuracy'], label='val accuracy')
    if 'filter_accuracy' in val_metrics.keys():
        for i, met_name in enumerate(['candidate_accuracy', 'filter_accuracy']):
            if met_name in val_metrics.keys():
                axs[1].plot(train_metrics['epoch'], train_metrics[met_name], color=f'C{i+1}', linestyle='dotted')
                axs[1].plot(val_metrics['epoch'], val_metrics[met_name], label=met_name, color=f'C{i+1}')
                axs[1].hlines(y=1, xmin=0, xmax=np.max(train_metrics['epoch']), color='red', linestyle='dashed')
                axs[1].set_ylabel('Accuracy', fontsize=14)
                axs[1].legend(fontsize=10)

    if 'ang_sep' in val_metrics:
        lonlat = np.array((dataset.candidate_grid.lon, dataset.candidate_grid.lat))
        pos1 = lonlat[:, :-1]
        pos2 = lonlat[:, 1:]
        ang_seps = geometry.angular_separation(pos1=pos1, pos2=pos2)
        average_cand_sep = np.mean(ang_seps)

        axs[2].plot(train_metrics['epoch'], np.array(train_metrics['ang_sep']), label='train', color='black', linestyle='dotted')
        axs[2].plot(val_metrics['epoch'], np.array(val_metrics['ang_sep']), label='val')
        axs[2].set_ylabel('Angular separation \n (deg)', fontsize=14)
        axs[2].set_xlabel('Epoch')
        axs[2].hlines(y=average_cand_sep/units.deg, xmin=0, xmax=np.max(train_metrics['epoch']), label='average candidate sep', color='red', linestyle='dashed')
        axs[2].legend(fontsize=10)

    for ax in axs:
        ax.grid(True, alpha=.5)

    fig.tight_layout()
    fig.savefig(run_paths.figures / 'loss_and_metrics_history.png')    
    
    if 'unique_candidates' in val_metrics:
        cand_ids, _ = np.unique(dataset.actions.detach().numpy(), return_counts=True)
        total_cand_diversity = len(cand_ids)/dataset.num_actions
        fig, ax = plt.subplots()
        ax.plot(train_metrics['epoch'], train_metrics['unique_candidates'], label='train', color='grey', alpha=.5, linestyle='dotted')
        ax.plot(val_metrics['epoch'], val_metrics['unique_candidates'], label='val')
        ax.set_ylabel('Unique candidates \n (normalized by total number of candidates)', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.hlines(y=total_cand_diversity, xmin=0, xmax=np.max(train_metrics['epoch']), label='dataset-wide unique candidate visit', color='black', linestyle='dotted')
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(run_paths.figures / 'unique_candidates_history.png')

    if 'lr' in train_metrics.keys():       
        fig, ax = plt.subplots()
        ax.grid(True, alpha=.5)
        ax.plot(train_metrics['epoch'], train_metrics['lr'])
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('LR', fontsize=14)
        fig.tight_layout()
        fig.savefig(run_paths.figures / 'lr_steps.png')

    # OTHER METRICS
    metrics_ya_plotted = ['accuracy', 'ang_sep', 'loss', 'unique', 'lr', 'unique']
    other_metrics = [item for item in val_metrics.keys() if not any(sub in item for sub in metrics_ya_plotted) and item != 'epoch']
    i = 0
    fig, ax = plt.subplots()
    for key in other_metrics:
        ax.plot(val_metrics['epoch'], val_metrics[key], label='val ' + key, color=f"C{i}")
        ax.plot(train_metrics['epoch'], train_metrics[key], color=f"C{i}", linestyle='dotted')
        i += 1
    ax.grid(True, alpha=.5)
    ax.legend()
    ax.set_xlabel('Epoch', fontsize=14)
    fig.tight_layout()
    fig.savefig(run_paths.figures / 'val_metrics.png')
    
def plot_bin_membership(dataset, fig_outdir):
    # Plot bin membership for fields in ra vs dec
    colors = [f'C{i}' for i in range(7)]
    for i, (bin_id, g) in enumerate(dataset._df.groupby('bin')):
        plt.scatter(g.ra, g.dec, label=bin_id, color=colors[i%len(colors)], s=1)
    plt.title("Fields in train data, colored by bin membership")
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(fig_outdir / 'train_data_fields_dec_vs_ra.png')

def plot_global_feature_distributions(dataset, fig_outdir):
    states = dataset.states.T
    ncols = 5
    nrows = len(dataset.global_feature_names) // ncols + 1
    fig = plt.figure(figsize=(ncols * 4, nrows * 3))

    for i, feat_name in enumerate(dataset.global_feature_names):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.hist(states[i])
        ax.set_title(f"{feat_name}")
    fig.tight_layout()
    fig.savefig(fig_outdir / 'train_global_feature_distributions.png')
    
def plot_candidate_feature_distributions(dataset, fig_outdir):
    ncols = 5
    nrows = len(dataset.candidate_feature_names) // ncols + 1
    fig = plt.figure(figsize=(ncols * 4, nrows * 3))

    cs_np = dataset.candidate_states.cpu().numpy()
    sm_np = dataset.candidate_sentinel_mask.cpu().numpy()

    for i, feat_name in enumerate(dataset.candidate_feature_names):
        ax = fig.add_subplot(nrows, ncols, i+1)
        feat = cs_np[..., i].copy()
        feat[sm_np[..., i]] = np.nan  # mask sentinels
        
        with np.errstate(all='ignore'):
            means = np.nanmean(feat, axis=1)                 # (T,)
            stds  = np.nanstd(feat,  axis=1)                 # (T,)
        
        # Optional: hide timestamps where every candidate is sentinel (means is NaN)
        keep = np.isfinite(means)
        x = np.arange(len(means))
        
        ax.plot(x[keep], means[keep])
        ax.fill_between(x[keep], (means - stds)[keep], (means + stds)[keep], alpha=0.3)
        ax.set_title(feat_name)
        ax.set_xlabel('timestep')
        ax.set_title(f"{feat_name}")
    fig.suptitle(f" Candidate features", fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_outdir / 'train_candidate_feature_distributions.png')
    

    # most_common_bin = dataset._df['bin'].mode()[0]
    # most_common_bin_states = dataset.bin_states[:, most_common_bin, :].detach().numpy()
    # bin_states = most_common_bin_states.T
    # for i, feat_name in enumerate(dataset.bin_feature_names):
    #     ax = fig.add_subplot(nrows, ncols, i+1)
    #     ax.hist(bin_states[i])
    #     ax.set_title(f"{feat_name}")
    # if dataset.hpGrid.is_azel: action_str = "(az, el)"
    # else: action_str = "(ra, dec)"
    # fig.suptitle(f" Train features for most common HEALpix bin (bin {most_common_bin}, {action_str} = {dataset.hpGrid.lon[most_common_bin]:.1f}, {dataset.hpGrid.lat[most_common_bin]:.1f})", fontsize=16)
    # fig.tight_layout()
    # fig.savefig(fig_outdir / 'train_bin_feature_distributions.png')
    
    