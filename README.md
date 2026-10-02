# DSAN6600 Project: Test whether earnings call transcripts predict post earnings stock volatility beyond the company's history

## Summary
In this project, I want to expand on the research from "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts" by Ding Yu, Zhuo Liu, Hangfeng He. Their research used text embedding models to predict future volatility of the company's stock. They concluded that transcript based models do not perform better than simply using the company's average past volatility. Additionally, they found that the transcript embeddings mainly encode which company is speaking.

So far, the company baseline explains only 9% of the variance in volatility. The provided embeddings identify the company with almost perfect accuracy, and a linear probe on them finds no signal beyond the baseline.

In the future, I will train an MLP head on the embeddings to see if, when company identifiers are stripped from the embeddings, there are signals of the movement of the company's stock (volatility). 


## Check in 1
- Write up: [check-in-1.md](check-in-1.md)
- Data Processing and EDA: [notebooks/EDA.ipynb](notebooks/EDA.ipynb)
- Data access: [DATA.md](DATA.md)
- Progress video: [link](...)

## Reproducing Results
1. Download the data into `data/` following [DATA.md](DATA.md)
2. Run `notebooks/EDA.ipynb` top to bottom

## References
## Attribution
Data and baseline from Yu et al. (2025), *Same Company, Same Signal* ([paper](https://aclanthology.org/2025.findings-acl.946/), [code/data](https://github.com/piqueyd/Same-Company-Same-Signal))