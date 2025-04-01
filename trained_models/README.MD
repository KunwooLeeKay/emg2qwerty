Put model configuration, model performance, and checkpoint link here
Epoch limit = 100, generic model only


### Kunwoo ###
1. Baseline - Causal (no change in architecture, only changed window padding)
- model checkpoint : https://drive.google.com/file/d/1h5Qu18zS3yhjNRHTwMf3MSnmW6_rwQIV/view?usp=share_link

2. TDS encoder + Attention decoder (No change to encoder)
- model architecture : https://drive.google.com/file/d/1p1Rjsq2it5xOEp8DYc84-WpSoL3JI0Yv/view?usp=share_link
- model checkpoint : Training!

### Chaeeun ### 

### Dhivya ### 

### Pushkar ###



### Results ### <- please add your model here
| Models         | Decoding | Language Model | Input Format | val CER | test CER | val loss | test loss | val IER | test IER | val DER | test DER | val SER | test SER |
|----------------|----------|----------------|--------------|---------|----------|----------|-----------|---------|----------|---------|----------|---------|----------|
| Baseline - Acausal | Greedy   | None           | Unchanged    | 55.57   | 55.38    | -        | -         | -       | -        | -       | -        | -       | -        |
| Baseline - Causal  | Greedy   | None           | Unchanged    | 24.9828 | 58.3192  | 0.9406   | 2.4479    | 6.74163 | 22.9193  | 2.9231  | 0.5936   | 15.3181 | 34.8062  |



