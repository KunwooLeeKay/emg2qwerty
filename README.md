# Enhancing Immersive Typing Experience via sEMG-Based Input Mapping with Hand Animation

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input, while optionally generating realistic hand animations to enhance immersion in virtual environments. 

> TODO - Update the project title at the end.

## Baseline

We reference [Meta's Github repo - emg2qwerty](https://github.com/facebookresearch/emg2qwerty), which uses the Meta emg2qwerty dataset - the largest publicly available sEMG-to-keyboard dataset. Below is our reproduced baseline:

| User       | Val CER (Greedy) | Test CER (Greedy) | Val CER (Beam Search) | Test CER (Beam Search) |
|------------|------------------|-------------------|------------------------|-------------------------|
| User0      |                  | 20.57%            |                        | 15.04%                  |
| User1      |                  | 10.32%            |                        | 6.18%                   |
| User2      |                  | 8.41%             |                        | 5.08%                   |
| User3      |                  | 8.93%             |                        | 4.74%                   |
| User4      |                  | 7.91%             |                        | 3.90%                   |
| User5      |                  | 5.81%             |                        | 3.16%                   |
| User6      |                  | 14.06%            |                        | 8.83%                   |
| **Average**|                  | **10.86%**        |                        | **6.71%**               |

> TODO - @Dhivya Sreedhar, do Beam Search results come with or without the LM module (I see the default is without LM)? Are they from generic or personalized models?

## Experiments

### Long list of ideas:

- [ ] **Experiment with data preprocessing and feature extractors**: 
  - [x] Modify the data loader to use past-only data for training  
  - [ ] Run training with new band-pass filters (40 Hz high-pass and 500 Hz low-pass)
  - [ ] Play with data pre-processing and augmentations 
- [ ] **Experiment with the model**: 
  - [ ] Incorporate transformer encoder / conformer / etc. 
- [ ] **Experiment with the inference**:
  - [ ] Incorporate a new LM module (explore character- vs word-level LM module + check gpt-2 style modules on top)
  - [ ] Include test results for online inference
- [ ] **Experiment with hand animation (if time permits)**:
  - [ ] Explore and incorporate the [emg2pose](https://github.com/facebookresearch/emg2pose) dataset  

> TODO - Add a summary of changes.

## Results

| Model Benchmark    | Val CER (Greedy) | Test CER (Offline, Greedy) | Test CER (Online, Greedy) | Val CER (Beam) | Test CER (Offline, Beam) | Test CER (Online, Beam) |
|--------------------|------------------|----------------------------|---------------------------|----------------|--------------------------|-------------------------|
| Meta baseline (reproduced)                         |         |    10.86%   |    n/a      |         |    6.71%    |    n/a     |
| Trained on past-only data @Kunwoo                  |         |         |         |         |         |         |
| New band-pass filters (40–500 Hz) @Pushkar         |         |         |         |         |         |         |
| Additional data augmentations (?)                  |         |         |         |         |         |         |
| Transformer Encoder v1 @Chaeeun                    |         |         |         |         |         |         |
| Transformer Encoder v2                             |         |         |         |         |         |         |
| Conformer v1                                       |         |         |         |         |         |         |
| New LM module @Dhivya                              |         |         |         |         |         |         |

Note. All results are averaged across subjects. Offline inference = Acausal (±900ms past / ±100ms future). Online inference = Causal (1000ms past-only)

> TODO - Add results + be clear about configs for each. Ideally, we need to have a separate config for each. 
  
## Acknowledgements

Special thanks to Meta AI for providing [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) and [emg2pose](https://github.com/facebookresearch/emg2pose) datasets.

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).
