# Typing Reinvented: Towards Hands-Free Input via sEMG

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input. 

## Results 

<p align="left">
  <a href="https://drive.google.com/file/d/1ptFljvxhDz1Og_-TxIwuATIsxz8HZnLf/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/View%20Final%20Report-blue?style=for-the-badge" alt="View Final Report">
  </a>
</p>

## Codebase

- [`notebooks`](./notebooks): Xxx

> TODO: Organize the folders and list them here. Also, tell how to run one experiment. 

## Long to-do of experiments

- [x] **Experiment with data preprocessing and feature extractors**: 
  - [x] Modify the data loader to use past-only data for training  
  - [x] Run training with new band-pass filters (40 Hz high-pass and 500 Hz low-pass)
- [x] **Experiment with the model**: 
  - [x] Incorporate transformer encoder / conformer / etc. 
- [x] **Experiment with the inference**:
  - [x] Incorporate a new LM module (explore character- vs word-level LM module + check gpt-2 style modules on top)
  - [x] Include test results for online inference
- [ ] **Experiment with hand animation (if time permits)**:
  - [ ] Explore and incorporate the [emg2pose](https://github.com/facebookresearch/emg2pose) dataset  
  
## Acknowledgements

Special thanks to Meta AI for providing [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) and [emg2pose](https://github.com/facebookresearch/emg2pose) datasets.

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).
