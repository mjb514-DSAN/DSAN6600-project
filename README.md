# Test whether earnings call transcripts predict post earnings stock volatility beyond the company's history

## Summary
In this project, I want to expand on the research from "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts" by Ding Yu, Zhuo Liu, Hangfeng He. Their research used text embedding models to predict future volatility of the company's stock. They concluded that transcript based models do not perform better than simply using the company's average past volatility. Additionally, they found that the transcript embeddings mainly encode which company is speaking.

### Check in 1 Findings
The company baseline explains only 9% of the variance in volatility. The provided embeddings identify the company with almost perfect accuracy, and a linear probe on them finds no signal beyond the baseline.

### Check in 2 Findings
I trained a regularized MLP head on the frozen embeddings under four input conditions (real, identity removed, identity only, no information). None beats predicting zero surprise on the test set, and identity only input does significantly worse. A positive control shows that the networks can learn company level volatility. This means that the embeddings carry who is speaking but nothing detectable about what changed in the call.

For the next check in, I will create chunked, Q&A aware embeddings with attention pooling, and a realized volatility target.


## Check in 1
- Write up: [check-in-1.md](check-in-1.md)
- Data Processing and EDA: [notebooks/EDA.ipynb](notebooks/EDA.ipynb)
- Data Access: [DATA.md](DATA.md)
- Progress Video: [Video](https://drive.google.com/file/d/1LsKrESfZDzSLU8VF6uK9L_SBvAyjoIJX/view?usp=sharing)

## Check in 2
- Write up: [check-in-2.md](check-in-2.md)
- Neural Baseline: [notebooks/neural_baseline.ipynb](notebooks/neural_baseline.ipynb)
- Progress Video: [Video-2](https://drive.google.com/file/d/1QfGRn-6OetblQi3Bs6OCwqhlaY3ypqFJ/view?usp=sharing)

## Reproducing Results
1. Download the data into `data/` following [DATA.md](DATA.md)
2. Run `notebooks/EDA.ipynb` top to bottom
3. Run `notebooks/neural_baseline.ipynb` top to bottom

## Attribution
Data and baseline from Yu et al. (2025), *Same Company, Same Signal* ([paper](https://aclanthology.org/2025.findings-acl.946/), [code/data](https://github.com/piqueyd/Same-Company-Same-Signal))