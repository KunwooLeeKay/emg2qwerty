# Enhancing Immersive Typing Experience via sEMG-Based Input Mapping with Hand Animation

This project explores the use of surface electromyography (sEMG) signals as an alternative input modality for typing. Our goal is to develop a system that maps wrist-based muscle activity to keyboard input, while optionally generating realistic hand animations to enhance immersion in virtual environments. It's a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## Baseline

We reference [Meta's Github repo - emg2qwerty](https://github.com/facebookresearch/emg2qwerty), which uses the Meta emg2qwerty dataset - the largest publicly available sEMG-to-keyboard dataset. Below is our reproduced baseline:

| User       | Test CER (Greedy) | Test CER (Beam Search) | Test Loss (Greedy) | Test Loss (Beam Search) |
|------------|-------------------|-------------------------|--------------------|--------------------------|
| User0      | 20.57%            | 20.57%                  | 1.17               | 1.17                     |
| User1      | 10.32%            | 10.32%                  | 0.65               | 0.65                     |
| User2      | 8.41%             | 8.41%                   | 0.51               | 0.51                     |
| User3      | 8.93%             | 8.93%                   | 0.47               | 0.47                     |
| User4      | 7.91%             | 7.91%                   | 0.37               | 0.37                     |
| User5      | 5.81%             | 5.81%                   | 0.38               | 0.38                     |
| User6      | 14.06%            | 14.06%                  | 0.84               | 0.84                     |
| **Average**| **10.86%**        | **10.86%**              | **0.63**           | **0.63**                 |

> *TODO - Check if the numbers above are correct (they're taken from the report).*
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
