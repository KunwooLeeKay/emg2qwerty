Put model configuration, model performance, and checkpoint link here
Epoch limit = 100, generic model only


### Kunwoo ###
1. Causal baseline (no change in architecture)
- model performance (Greedy Decoding)
| Model Name | Baseline - Acausal | Baseline - Causal |
|----------|----------|----------|
| Decoding | Greedy | Greedy |
|----------|----------|----------|
| Language Model | None | None |
|----------|----------|----------|
| val CER | Value 2  | 24.9828  |
| test CER | Value C  | - |
|----------|----------|----------|
| val loss | - | 0.9406 |
| test loss | - | 2.4479 |
|----------|----------|----------|
| val IER | - | 6.74163 |
| test IER | - | - |
|----------|----------|----------|
| val DER | - | 2.9231 |
| test DER | - | - |
|----------|----------|----------|
| val SER | - | 15.3181 |
| test SER | - | - |


| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value A  | Value B  | Value C  |



{'val_metrics': [{'val/loss': 0.9406100511550903,
                  'val/CER': 24.982807159423828,
                  'val/IER': 6.741629600524902,
                  'val/DER': 2.9231131076812744,
                  'val/SER': 15.318068504333496}], 
 'test_metrics': [{'test/loss': 2.44797682762146,
                   'test/CER': 58.319183349609375,
                   'test/IER': 22.919342041015625,
                   'test/DER': 0.5935865044593811,
                   'test/SER': 34.80625534057617}],
- model checkpoint : https://drive.google.com/file/d/1h5Qu18zS3yhjNRHTwMf3MSnmW6_rwQIV/view?usp=share_link


