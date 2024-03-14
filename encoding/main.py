import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_config, parse_arguments
from encoder import run_subcortical, run_cortical


if __name__ == '__main__':
    print('Running: ', __file__)

    # Configuration (config.yaml)
    config = load_config('config.yaml')

    default_subjects = config['subjects']
    subjects = parse_arguments(default_subjects)

    SSD_dir = Path(config['SSD_dir'])
    results_cortex_dir = Path(config['results_cortex_dir'])
    results_subcortex_dir = Path(config['results_subcortex_dir'])
    results_cortex_dir.mkdir(parents=True, exist_ok=True)
    results_subcortex_dir.mkdir(parents=True, exist_ok=True)

    alphas = config['alphas']
    cortex_sfreq = config['cortex']['sfreq']

    make_plots = config['make_plot']

    # Initialize alpha vectors
    acc_subcortex = np.zeros(len(subjects)) * np.nan
    acc_cortex = np.zeros(len(subjects)) * np.nan

    # Initialize best alpha vectors
    best_alphas_sub = np.zeros(len(subjects)) * np.nan
    best_alphas_subinv = np.zeros(len(subjects)) * np.nan
    best_alphas_cor = np.zeros(len(subjects)) * np.nan

    for idx, subject_id in enumerate(subjects):
        print(f'Running participant {subject_id}')

        response, score, lags, alpha_scores, best_alpha = run_subcortical(
            subject_id,
            SSD_dir,
            alphas=alphas,
            tmin=config['subcortex']['tmin'],
            tmax=config['subcortex']['tmax'],
            sfreq=config['subcortex']['sfreq'],
            invert=False
        )

        response_inv, score_inv, _, alpha_scores_inv, best_alpha_inv = run_subcortical(
            subject_id,
            SSD_dir,
            alphas=alphas,
            tmin=config['subcortex']['tmin'],
            tmax=config['subcortex']['tmax'],
            sfreq=config['subcortex']['sfreq'],
            invert=True
        )

        # Get average response and score between normal and inverted model
        response_sub = (response + response_inv) / 2
        score_sub = (score + score_inv) / 2
        acc_subcortex[idx] = score_sub
        best_alphas_sub[idx] = best_alpha
        best_alphas_subinv[idx] = best_alpha_inv

        # Run cortical model
        resonse_cor, score_cor, lags_cor, alpha_scores_cor, best_alpha_cor = run_cortical(
            subject_id,
            SSD_dir,
            alphas=alphas,
            tmin=config['cortex']['tmin'],
            tmax=config['cortex']['tmax'],
            sfreq=config['cortex']['sfreq']
        )

        acc_cortex[idx] = score_cor
        best_alphas_cor[idx] = best_alpha_cor

        # Save response waveforms
        np.save(results_subcortex_dir / f'{subject_id}', response_sub)
        np.save(results_cortex_dir / f'{subject_id}', resonse_cor)

        # ---------------- Plot ----------------
        if make_plots:
            figures_dir = Path(config['figures_dir'])
            figures_dir.mkdir(parents=True, exist_ok=True)

            fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [2, 1]})

            ax1 = axs[0, 0]
            ax2 = axs[0, 1]
            ax3 = axs[1, 0]
            ax4 = axs[1, 1]

            fig.suptitle(f'Participant {subject_id}')

            # ---------------- Subcortex subplots ----------------
            ax1.set_title(f'Score: {score_sub:.4f}')
            ax1.plot(lags, response_sub.T)
            ax1.set_xlabel('Lag (ms)')
            ax1.set_ylabel('Weights')
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(-np.max(np.abs([ymin, ymax])), np.max(np.abs([ymin, ymax])))

            ax2.set_title('Alpha tuning')
            ax2.semilogx(list(alpha_scores.keys()), list(alpha_scores.values()), '-o')
            ax2.semilogx(list(alpha_scores_inv.keys()), list(alpha_scores_inv.values()), '-o')

            # ---------------- Cortex subplots ----------------
            ax3.set_title(f'Score: {score_cor:.3f}')
            ax3.plot(lags_cor, resonse_cor.T)
            ax3.set_xlabel('Lag (ms)')
            ax3.set_ylabel('Weights')
            ymin, ymax = ax3.get_ylim()
            ax3.set_ylim(-np.max(np.abs([ymin, ymax])), np.max(np.abs([ymin, ymax])))

            ax4.set_title('Alpha tuning')
            ax4.semilogx(list(alpha_scores_cor.keys()), list(alpha_scores_cor.values()), '-o', color='darkblue')

            # ---------------- Save ----------------
            fig.savefig(figures_dir / f'{subject_id}_subcortex.pdf', bbox_inches='tight')
            plt.close(fig)

    # Store hyperparameter results and accuracies in a dataframe
    df = pd.DataFrame({
        'subject_id': subjects,
        'subcortex_accuracies': acc_subcortex,
        'subcortex_alphas': best_alphas_sub,
        'subcortex_alphas_invert': best_alphas_subinv,
        'cortex_accuracies': acc_cortex,
        'cortex_alphas': best_alphas_cor,
    })

    df.to_csv(results_subcortex_dir / 'results.csv')
