# Preprocessing and encoding analysis pipelines for the encoding of natural speech at the subcortical and cortical level

This repository contains the data pipelines described in *Data preprocessing* and *TRF modeling* in our paper entitled *Auditory encoding of natural speech at subcortical and cortical levels is not indicative of cognitive decline*, [doi:10.1523/eNeuro.0545-23.2024](https://doi.org/10.1523/ENEURO.0545-23.2024).

We share the code for reasons of transparency. The data—EEG and audio files—cannot be shared, but are available upon request. The code is adapted to our environment and data infrastructure and cannot be executed without adjustments.

## Structure

- **preprocessing/**: This folder contains the EEG and audio file preprocessing pipelines.
  - **preprocessing/audio/**: Within this folder, there is a subfolder called `an_model` which contains the scripts and an additional environment file (`an_model_environment.yml`) needed to generate the speech features (auditory nerve rates) for subcortical analyses.

- **encoding/**: This folder contains the pipeline for the encoding analyses.
  
  - Please note that the TRF model is referred to in the scripts as "encoding model" and "encoder," and the regularization parameter λ is referred to as "alpha."

![Encoding models and evoked responses obtained through our pipeline.](responses.png)

## Environment setup

The `environment.yml` file included in this repository contains the specifications for the conda environment used in this project. Please note that the code is adapted to our environment and data infrastructure and cannot be executed without adjustments.

## Acknowledgments

We would like to thank Shan et al. (2024) as we benefited from their published analysis pipeline when generating speech features from auditory nerve rates, GitHub repository [Music_vs_Speech_abr](https://github.com/maddoxlab/Music_vs_Speech_abr) and publication [doi:10.1038/s41598-023-50438-0](https://www.nature.com/articles/s41598-023-50438-0).
