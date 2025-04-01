Put model configuration, model performance, and checkpoint link here
Epoch limit = 100, generic model only


### Kunwoo ###
1. Baseline - Causal (no change in architecture, only changed window padding)

| Model Name     | Baseline - Acausal | Baseline - Causal |
| Decoding       | Greedy             | Greedy            |
| Language Model | None               | None              |
| Input Format   | Unchanged          | Unchanged         |
|----------------|--------------------|-------------------|
| val CER        | 55.57              | 24.9828           |
| test CER       | 55.38              | 58.3192           |
| val loss       | -                  | 0.9406            |
| test loss      | -                  | 2.4479            |
| val IER        | -                  | 6.74163           |
| test IER       | -                  | 22.9193           |
| val DER        | -                  | 2.9231            |
| test DER       | -                  | 0.5936            |
| val SER        | -                  | 15.3181           |
| test SER       | -                  | 34.8062           |

- model checkpoint : https://drive.google.com/file/d/1h5Qu18zS3yhjNRHTwMf3MSnmW6_rwQIV/view?usp=share_link


