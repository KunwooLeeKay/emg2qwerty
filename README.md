# Typing Reinvented: Towards Hands-Free Input via sEMG

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input, while optionally generating realistic hand animations to enhance immersion in virtual environments. 

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

## Results

> TODO - Add a link to the final report. 
  
## Acknowledgements

Special thanks to Meta AI for providing [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) and [emg2pose](https://github.com/facebookresearch/emg2pose) datasets.

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).
