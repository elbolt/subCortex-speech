import numpy as np
import pandas as pd
from pathlib import Path
from utils import default_subjects, parse_arguments
from encoder import run_subcortical, run_cortical


if __name__ == '__main__':
    print('Running: ', __file__)

    # Paths to my data directories
    SSD_dir = Path('/Volumes/NeuroSSD/Midcortex/data')
    results_cortex_dir = SSD_dir / 'EEG/TRF/weights/cortex'
    results_subcortex_dir = SSD_dir / 'EEG/TRF/weights/subcortex'
    results_cortex_dir.mkdir(parents=True, exist_ok=True)
    results_subcortex_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_arguments(default_subjects)

    alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

    # Accuracies
    acc_subcortex = np.zeros(len(subjects)) * np.nan
    acc_cortex = np.zeros(len(subjects)) * np.nan

    # Best alphas
    best_alphas_sub = np.zeros(len(subjects)) * np.nan
    best_alphas_subinv = np.zeros(len(subjects)) * np.nan
    best_alphas_cor = np.zeros(len(subjects)) * np.nan

    for idx, subject_id in enumerate(subjects):
        print(f'Running participant {subject_id}')

        # Run subcortical models (normal and inverted)
        response, score, alpha_scores, best_alpha, _ = run_subcortical(subject_id, SSD_dir, alphas=alphas)

        response_inv, score_inv, alpha_scores_inv, best_alpha_inv, _ = run_subcortical(
            subject_id,
            SSD_dir,
            alphas=alphas,
            invert=True
        )

        # Get average response and score betweeon normal and inverted model
        response_sub = (response + response_inv) / 2
        score_sub = (score + score_inv) / 2
        acc_subcortex[idx] = score_sub
        best_alphas_sub[idx] = best_alpha
        best_alphas_subinv[idx] = best_alpha_inv

        # Run the cortical model
        resonse_cor, score_cor, _, best_alpha_cor, alpha_scores_cor = run_cortical(
            subject_id,
            SSD_dir,
            alphas=alphas
        )

        acc_cortex[idx] = score_cor
        best_alphas_cor[idx] = best_alpha_cor

        # Save response waveforms
        np.save(results_subcortex_dir / f'{subject_id}', response_sub)
        np.save(results_cortex_dir / f'{subject_id}', resonse_cor)

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
