# Preprocessing and encoding analysis pipelines for the encoding of natural speech at the subcortical and cortical level

## Description

This repository contains the code for a manuscript yet to be published.
*[Info yet to follow.](https://u.rl)*

We share the code for reasons of transparency. The data—EEG and audio files—cannot (yet) be shared. The code is adapted to our environment and data infrastructure, it cannot be executed without adjustments.

* The required modules that we have used to execute our code can be found in the `environment.yml` file.
* All data paths would have to be adapted to your own data infrastructure.


## Structure

* The EEG and audio file preprocessing pipelines are located in the `preprocessing` folder.
* In `preprocessing/audio` there is a subfolder called `an_model` which contains the scripts and an additional environment (`an_model_environment.yml`) that I needed to generate the speech features (auditory nerve rates) for subcortical analyses.
* The pipeline for the encoding analyses is located in the `encoding` folder.
* Please note that the TRF model is referred to in the scripts as "encoding model" and "encoder" and the regularization parameter λ as "alpha"


## Acknowledgments

We would like to thank Shan et al. (2022) as we benefited from their published analysis pipeline when generating speech features from auditory nerve rates, GitHub repository [Music_vs_Speech_abr](https://github.com/maddoxlab/Music_vs_Speech_abr) and publication [doi: 10.1101/2022.10.14.512309](https://doi.org/10.1101/2022.10.14.512309).
