# Enhancing Immersive Typing Experience via sEMG-Based Input Mapping with Hand Animation

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input, while optionally generating realistic hand animations to enhance immersion in virtual environments. 

> TODO - Update the project title at the end.

## Baseline

We reference [Meta's Github repo - emg2qwerty](https://github.com/facebookresearch/emg2qwerty), which uses the Meta emg2qwerty dataset - the largest publicly available sEMG-to-keyboard dataset. Below is our reproduced baseline:

| User       | Val CER (Greedy) | Test CER (Greedy) | Val CER (Beam)         | Test CER (Beam)         |
|------------|------------------|-------------------|------------------------|-------------------------|
| User0      |    17.96%         | 20.57%            |     13.71%                  | 15.04%                  |
| User1      |    8.39%             | 10.32%            |  6.11%                      | 6.18%                   |
| User2      |     8.16%             | 8.41%             |  6.22%                      | 5.08%                   |
| User3      |      9.54%            | 8.93%             |  6.50%                      | 4.74%                   |
| User4      |      7.57%            | 7.91%             |  5.23%                      | 3.90%                   |
| User5      |      7.15%            | 5.81%             |   5.51%                     | 3.16%                   |
| User6      |       15.19%           | 14.06%            |     12.3%                   | 8.83%                   |
| **Average**|       **10.42%**          | **10.86%**        |     **7.94%**                  | **6.71%**               |

**Note**: We use a personalized model. Beam Search results are with the LM module. 

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
  - [x] Include test results for online inference
- [ ] **Experiment with hand animation (if time permits)**:
  - [ ] Explore and incorporate the [emg2pose](https://github.com/facebookresearch/emg2pose) dataset  

> TODO - Add a summary of changes.

## Results

| #    | Model Benchmark                                                  | Val CER (Greedy) | Test CER (Offline, Greedy) | Test CER (Online, Greedy) | Val CER (Beam) | Test CER (Offline, Beam) | Test CER (Online, Beam) |
|------|------------------------------------------------------------------|------------------|-----------------------------|----------------------------|----------------|----------------------------|---------------------------|
| **1**  | Meta **generic** baseline (from Meta's paper)                  | 55.57% ± 4.40     | 55.38% ± 4.10               | n/a                        | 52.10% ± 5.54   | 51.78% ± 4.61              | n/a               |
| **2**  | Meta **personalized** baseline (from Meta's paper)             | 11.39% ± 4.28     | 11.28% ± 4.45               | n/a                        | 8.31% ± 3.19    | 6.95% ± 3.61               | n/a        |
| **3**  | Meta **personalized** baseline (reproduced)                    | 10.42%            | 10.86%                      | n/a                        | 7.94%          | 6.71%                      | n/a             |
| **4**  | Causal **generic** (trained on past-only data) @Kunwoo         | 24.98%            | 58.32%                      | ~20% (tbc)                 |                |                            |               |
| **5**  | New band-pass filters (40–500 Hz) @Pushkar                     | 27.06%           |   56.95%                   |                            |                |                            |                     |
| **6**  | Transformer Encoder v1 (d_model = 4) @Chaeeun                  | 24.64%            |                             |                            |                |                            |               |
| **7**  | Transformer Encoder v2 (d_model = 768) @Chaeeun                | 32.55%            |                             |                            |                |                            |                     |
| **8**  | Transformer Encoder v3 (d_model = 768, FF = 256) @Chaeeun      | 37.34%            |                             |                            |                |                            |          |
| **9**  | Transformer Decoder v1 @Kunwoo                                | 41.20%            |                             |                            |                |                            |          |
| **10** | Conformer v1  @Kunwoo                                                  |                  |                             |                            |                |                            |                  |
| **11** | Model [pick a number] + new LM module v1 @Dhivya               | n/a              | n/a                         | n/a                        |                |                            |                     |


**Note**: 
- 100 epochs of training.
- For each run, we use a generic model, but we reproduced the Meta baseline using a personalized model. 
- Offline inference = Acausal (±900ms past / ±100ms future).
- Online inference = Causal (1000ms past-only).
- Beam Search results are with the LM module. A default one is a 6-gram LM. 
- Val/test results are averaged across subjects.

> TODO - Add results + have a separate config for each run. 
  
## Acknowledgements

Special thanks to Meta AI for providing [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) and [emg2pose](https://github.com/facebookresearch/emg2pose) datasets.

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).
