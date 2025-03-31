# Enhancing Immersive Typing Experience via sEMG-Based Input Mapping with Hand Animation

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input, while optionally generating realistic hand animations to enhance immersion in virtual environments. It's a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## Baseline

We reference [Meta's Github repo - emg2qwerty](https://github.com/facebookresearch/emg2qwerty), which uses the Meta emg2qwerty dataset - the largest publicly available sEMG-to-keyboard dataset. Below is our reproduced baseline:

| User       | Test CER (Greedy) | Test CER (Beam Search) | Test Loss (Greedy) | Test Loss (Beam Search) |
|------------|-------------------|-------------------------|--------------------|--------------------------|
| User0      | 20.57%            | 15.04%                  | 1.17               | 1.174                     |
| User1      | 10.32%            | 6.18%                  | 0.65               | 0.651                     |
| User2      | 8.41%             | 5.08%                   | 0.51               | 0.514                     |
| User3      | 8.93%             | 4.74%                   | 0.47               | 0.469                     |
| User4      | 7.91%             | 3.90%                   | 0.37               | 0.365                     |
| User5      | 5.81%             | 3.16%                   | 0.38               | 0.377                     |
| User6      | 14.06%            | 8.83%                  | 0.84               | 0.73                     |
| **Average**| **10.86%**        | **6.709%**              | **0.63**           | **0.612**                 |

> *TODO - Check if the numbers above are correct (they're taken from the report) + maybe drop info about loss and show val vs test.*
> 
> *TODO - Add a disclaimer about the testing procedure (model (generic or personalized),  size of the test set, etc.) - see Table 3 in Meta report.*

## Experiments

### Long list of ideas:

- [ ] **Experiment with data preprocessing and feature extractors**: 
  - [ ] Modify the data loader to use past data only (suitable for online inference)  
  - [ ] Run training with new band-pass filters (40 Hz high-pass and 500 Hz low-pass)
- [ ] **Experiment with the model**: 
  - [ ] Incorporate transformer encoder
  - [ ] Incorporate LM module
- [ ] **Experiment with the inference**: 
  - [ ] Run online inference
- [ ] **Experiment with hand animation (if time permits)**:
  - [ ] Explore and incorporate the [emg2pose](https://github.com/facebookresearch/emg2pose) dataset  

> *TODO - Add a summary of changes.*

## Results

> *TODO - Add a table with results after final experiments.*
  
## Acknowledgements

Special thanks to Meta AI for providing [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) and [emg2pose](https://github.com/facebookresearch/emg2pose) datasets.
