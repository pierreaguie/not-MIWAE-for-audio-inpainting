# not-MIWAE for Audio Declipping

This repository contains an implementation of not-MIWAE, based on [Ipsen et al. "not-MIWAE: Deep Generative Modelling with Missing not at Random Data"](https://arxiv.org/abs/2006.12871), applied to the task of audio declipping. Technical details on this implementation and experimental results can be found in [this report](assets/report/PGM_Report.pdf).

## Usage

This implementation was written in Python 3.10.7 and uses PyTorch. To install the required Python libraries using pip, run

```bash
pip install -r requirements.txt
```

To train a not-MIWAE model, run
```bash
python train.py --lr 1e-3 --batch_size 64 --nepochs 1000
```

## Repository structure

This repository contains the following files:

```bash
├── README.md
├── assets
│   ├── audio                                  # Examples reconstructions of clipped audio files
│   │   ├── ...
│   ├── plots                                  # Figures used in the report
│   │   ├── ...
│   └── report
│       └── PGM_Report.pdf                     # Report presenting the project
├── data                                       # Folder containing the datasets
│   ├── ...
├── experiment_clipping_percentage.py          # Experiment to study the impact of the proportion of clipped values on reconstruction quality
├── experiment_latent_dim.py                   # Experiment to study the impact of latent dimension on reconstruction quality
├── get_imputations.py                         # Scipt to impute clipped audio using a trained not-MIWAE model
├── requirements.txt
├── src
│   ├── datasets.py                            # Data processing functions
│   ├── models.py                              # not-MIWAE and missing model modules
│   ├── train_test.py                          # Training and validation functions
│   └── utils.py                               # Utility function for audio signal processing
├── test_audio_quality.py                      # Computes spectral distance between original audio and not-MIWAE reconstruction
└── train.py                                   # Script to train a not-MIWAE model
```

## Citation

This project is based on the following paper:

```bash
@inproceedings{ipsen2021notmiwae,
title={not-{\{}MIWAE{\}}: Deep Generative Modelling with Missing not at Random Data},
author={Niels Bruun Ipsen and Pierre-Alexandre Mattei and Jes Frellsen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=tu29GQT0JFy}
}
```


