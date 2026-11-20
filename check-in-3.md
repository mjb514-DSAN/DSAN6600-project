## Check in 3: Attention Extension

### Summary of Results
Check in 2 found that the paper's single embedding per call couldn't predict post earning volatility surprise. For this check in, I tested whether that representation was the bottleneck. 94.7% of calls exceed the embedding model's token limit, and one vector averages a whole call into a single point. I replaced it with ~150 word chunk embeddings pooled by either a masked mean (model B) or learned attention (model C). On raw volatility, both chunked models significantly beat the check in 2 baseline, but a shuffle test shows that signal is entirely company identity. On the surprise,  no model beats predicting zero. Attention weights were near uniform, so attention added nothing over mean pooling. 

### Adjustments After Check in 2
Thanks to helpful feedback from the TAs, there were a few adjustments that needed to be made from my work in check in 2 before I continued: epoch cap and independence.

To deal with epoch caps, I raised max epochs to 300, with a patience of 20, plus an automatic retry with 2x the cap if the best epoch was greater than or equal to 90% of the cap. The retry never triggered, so the cap is no longer binding. The best epochs were 0-60.

For independence, before I treated each call as independent, but calls from the same company are not independent of each other. Now, every confidence interval mentioned is a company-clustered paired bootstrap. I resampled companies, not individual calls, so all of a company's quarters stay together, and every model is scored on the same draws so comparisons between models are paired. Rescoring check in 2 under the clustered bootstrap makes the random ticker results borderline (CI of [-0.0149, -0.0000]). My check in 2 claim that the random ticker model is worse than predict zero is weaker than I reported, but the rest of the results remain the same.


### Why this Extension
Check in 2's MLP on the paper's single OpenAI vector per call didn't beat predicting zero on the volatility surprise, but it could learn raw volatility. This left me with two possible explanations. The text had nothing beyond company history, or the representation was the bottleneck. Using a single vector per call could be throwing the signal away before the model ever saw it.

During this check in, I wanted to test the second explanation. If my results indicate that a richer representation still finds nothing, the idea that the text had nothing beyond company history has more validity. Two problems with the single vector approach motivated the extension.

First, the calls didn't fit in the embedding. The paper's embedding model accepts at most 8,191 tokens. However, 94.7% of calls exceeded that limit. The median call is 11,539 tokens. This means that if calls were truncated at the limit, then a median of 29.7% of each over the limit call would not be seen. The Q&A starts before the limit in 97.5% of calls, so truncation would keep the prepared remarks and early Q&A but cut the rest of the questions. It's not clear from the paper how long calls were handled. Instead of truncating, I split each call into ~150 word chunks and embedded every chunk, so no part of the call is lost.

The second problem was that one vector averages everything. Even without truncation, one vector compresses about 60 chunks' worth of content into a single point. This could mean that important information would get diluted by less important information (hellos, thank yous, routine recaps). From what I know about calls, a few moments dictate the market movement, so diluting these specific moments would hurt the model. To fix this, I used attention pooling. The model learns which chunks to weight, rather than weighting all of them equally.

The Q&A is important to keep because prepared remarks typically are scripted and vetted in advance. However, the Q&A is unscripted which may provide stronger signals. Truncating would cut the later Q&A.

I replace the single vector with ~150 word chunk embeddings from a local encoder, pooled by either a masked mean (model B) or a learned attention query (model C). This led me to have three hypotheses going into this phase of research.
- Hypothesis 1: chunked embeddings recover call specific signal that the single vector missed (testing model B vs A).
- Hypothesis 2: attention pooling beats a plain average of the same chunks (model C vs model B)
- Hypothesis 3: if there is signal, it lives in the Q&A, not in the prepared remarks.


### The Extension

#### Pipeline
The pipeline works like this: transcript → clean → split remarks / Q&A → ~150-word chunks → embed each chunk (local) → sequence of chunk vectors per call → pooling (mean or attention) → small MLP head → predicted surprise

#### Cleaning and Finding where Q&A starts
For cleaning, I deduplicated the transcript text (check in 1), and fixed encoding errors with ftfy. Check in 2's embeddings came from the paper and didn't use my cleaned text, but the embeddings in this check in use the cleaned text.

I needed to find where the Q&A starts in order to know what truncation would cut, and for testing the signal in Q&A versus prepared remarks (hypothesis 3). Simple text markers like "question and answer" failed as they showed up at a median position of .005 which was typically the intro. I ran through four versions, checking each version's failures by reading the transcripts.

First, I used the "first question" phrase. In some cases, this would occur earlier than the actual Q&A (NKE and TGT for example). Second, I used the operator's line that introduces questions, but this missed operators who didn't say 'question' (AEP's call). Next, I used any operator line after the intro, but some calls didn't have a labeled operator (AMD's call). The final version used the earliest of an operator line or a "first/next question" line, ignoring those being said in the first 10% of the call. As a result, 1,796 of the 1,800 calls split. Doubtful splits were flagged, and hypothesis 3 tests only using cleanly split calls (356 of the 360 test calls).

#### Chunking and Encoder
I used 150 word chunks, with a minimum of 20 words. I chose 150 because the encoder reads at most 256 tokens, so 200 word chunks would exceed that. At 150 words, the 99th percentile is 221 tokens, so no chunks were cut. This resulted in 117,169 total chunks, with a median of 63 chunks per call.

For the encoder, I used all-MiniLM-L6-v2, 384-d, normalized, run locally. I used a local model rather than the OpenAI API so that I could have full control over chunking, didn't have to deal with API costs, and so that transcripts could stay on my machine because the transcripts are for academic use only.

As a sanity check, I found that neighboring chunks in the same call have cosine similarity of .519 versus .299 for random pairs. This means the embeddings capture content and the rows are aligned. The chunk vectors were standardized using means and SDs from the training chunks only to prevent leakage.

### Models

#### The Three Models
Each call becomes a variable-length sequence of chunk vectors. Batches are padded with `pad_sequence`, and a mask marks real vs padded chunks.

Model A: The check in 2 baseline model was an MLP on the paper's single 3,072-d OpenAI vector, unchanged. The test predictions were rescored with the new bootstrap method.

Model B: Using masked mean pooling. Project each chunk 384 → d with a linear layer, then average only the real chunks.

Model C: Attention pooling. Using the same value projection, plus a key projection and one learned query vector. Padded chunks were set to negative infinity before the softmax so they get zero weight. Each chunk's score is its key dotted with the learned query, divided by the square root of d; a softmax over the score gives the attention weights. The call vector is the attention-weighted sum of the values.

B and C share the same head: Dropout → Linear → ReLU → Dropout → Linear. B and C are identical except for the pooling step so that the comparison between B and C can isolate whether attention helps. Model B changes both the encoder and the chunking at once compared to model A so hypothesis 1 tests the combined change.

#### Training
Training used MSE loss, Adam optimizer, batch size 32, weight decay of .01. I used early stopping on validation MSE with a patience of 20 and a max of 300 epochs. Both models overfit a tiny batch to ~0 loss in 500 steps, a check that the pipeline can learn.

#### Tuning and Locking
I tuned both models on validation only, using a grid over the projection size d, hidden size, dropout, and learning rate. With a single seed, the top configurations often had their best validation epoch at 0-2, essentially before any training, which meant the grid was mostly selecting noise. To fix this, I re-ran the top configurations across 3 seeds and picked the one with the lowest mean validation MSE. The locked configurations were d = 32 for B and d = 16 for C. Both models had hidden size 4, dropout .7, and learning rate 1e-3. Each locked model was trained as a 5 seed ensemble and evaluated on the test set once.

### Protocol
To ensure that comparisons across models was fair, I held four things fixed: same target, same splits, same baseline, and same evaluation for all three models.

All 3 models used `surprise3_q` which is the paper's 3 day log volatility after the call, minus the company's own past average, minus the calendar quarter mean. Additionally, all three models used a training set of 2019-2021, validation set of 2022, and test set of 2023. Additionally, all three models were compared to the same baseline model of the predict zero model. The predict zero model predicts that this call is as volatile as the company usually is. The baseline model had a test MSE of .5254. Every model's results are reported as its gains over the baseline. Lastly, model A's saved check in 2 test predictions were rescored with the same company-clustered paired bootstrap as models B and C. This way, differences between models are paired.

### Results
All three models were evaluated on the same 2023 test set, and gains are measured relative to the predict zero model for surprise volatility, or the global mean model for raw volatility.

#### Surprise Volatility
| Model | Test MSE | Gain vs. predict-zero | 95% CI (company-clustered) |
|---|---|---|---|
| Predict zero | 0.5254 | — | — |
| A: Check-in 2 single vector | 0.5266 | −0.0011 | [−0.0067, 0.0041] |
| B: Chunks + mean pooling | 0.5260 | −0.0006 | [−0.0073, 0.0059] |
| C: Chunks + attention pooling | 0.5265 | −0.0011 | [−0.0061, 0.0038] |

No model beats predicting zero as every confidence interval includes 0. The difference of B-A is [-.0049, .0062], and C-A is [-.0052, .0059]. The largest upper bound is +.0059, about 1.1% of predict zero MSE. If the chunked text helps at all, it improves MSE by at most roughly 1%.

#### Raw Volatility
| Model | Test MSE | Gain vs. global mean | 95% CI (company-clustered) | Share of gap closed |
|---|---|---|---|---|
| Global mean | 0.6333 | — | — | 0% |
| A: Check-in 2 single vector | 0.6124 | +0.0209 | [−0.0073, 0.0486] | 23% |
| B: Chunks + mean pooling | 0.5782 | +0.0551 | [0.0288, 0.0880] | 60% |
| C: Chunks + attention pooling | 0.5761 | +0.0572 | [0.0297, 0.0892] | 62% |
| Company baseline | 0.5411 | +0.0922 | — | 100% |

Model A doesn't significantly beat the global mean, but models B and C do. B-A produces a confidence interval of [.012, .060], and C-A produces a confidence interval of [.014, .061]. The extension clearly helped for raw volatility. However, neither model reaches the company baseline so predicting each company's average still beats the text models.

Overall, the chunked representation is better at encoding raw volatility, but not call-specific information needed for surprise volatility. What the chunked models learned on raw volatility will be examined in the failure analysis.

### Ablations
I made three comparisons, each changing one thing. The pooling method (hypothesis 2), which part of the call the model sees (hypothesis 3), and which label it predicts.

#### Attention vs Mean
Models B and C are identical except for pooling. For predicting surprise volatility, C-B gave a confidence interval of [-.0035, .0029], meaning that the two models were indistinguishable. For raw volatility, C-B produced a confidence interval of [-.0039, .0085], meaning there was no significant difference. The claim in hypothesis two is not supported. Attention gave no measurable benefit over a plain average. I will explain this more in the failure analysis.

#### Q&A only vs Remarks only
To test hypothesis 3, model C was retrained on only the Q&A chunks or only the remarks chunks, and was scored on the 356 test calls that had a clean split.

For surprise volatility, the Q&A only compared to the predict zero baseline scored a confidence interval of [-.0050, .0045], remarks only compared to the baseline scored a CI of [-.0077, .0019], and the full call scored against the baseline had a confidence interval of [-.0060, .0040]. No version significantly outperformed the predict zero baseline. 

Q&A - remarks produced a confidence interval of [-.0006, .0057], leaning towards Q&A having stronger signal, but the confidence interval still includes zero so results were not significant. 

For raw volatility, all three versions significantly outperformed the global mean baseline model. The full call was +.058 better than baseline, Q&A only was +.041 better than baseline, and remarks only was +.036 better than baseline. Comparing Q&A to remarks only was not significant (CI of [-.0062, +.0178]). However, comparing Q&A to the full call showed that the full call was significantly better (confidence interval of [-.0279, -.0073]). 

Each half carries enough to identify the company, and more text helps. Overall, the claims in hypothesis 3 were not supported. The Q&A lean is in the predicted direction, but is not significant. 

#### The realized volatility surprise
The paper's label is built from a demeaned 3 day log volatility. I built an alternative from raw returns, with the same company history and quarter adjustments in order to check that the null isn't an artifact of the label choice. I found that the two labels are related (correlation of .859). The realized volatility surprise is much less noisy, with a predict zero MSE of .378 vs .525 for the paper's label.

I reran the models on the realized volatility surprise, but neither B nor C was significantly stronger than the baseline model, and there was no significant difference between models C and B. The cleaner target didn't change my conclusions.

### Failure Analysis
On raw volatility, the chunked models learn something, and on the surprise volatility they learn nothing. During the failure analysis, I wanted to see what they learned, and why it didn't transfer.

#### Shuffle Test
To see what the models actually learned, I used a shuffle test. Models B and C were trained normally on raw volatility. At test time, each call's chunks were replaced with either another 2023 call from the same company, or a random call from a different company. If it only recognizes the company, then the same company replacement would change nothing and the different company replacement would break it. If the model used this call's content, swapping in another quarter would hurt it.

For model B, the true call test MSE was .5782, same company replacement MSE was .5776, and different company replacement was .6533. For model C, the true call test MSE was .5761, same company replacement MSE was .5758, and different company replacement was .6489. 

This indicates that the models don't need this specific call because any call from the same company works equally well. On the other hand, a different company's call makes it worse than the global mean because the model predicts the wrong company's typical volatility. This means that all of the raw volatility signal is company identity. The surprise subtracts each company's own history, which removes that, so nothing is left to learn. This is consistent with the paper's "same company, same signal" finding.

#### Attention is near uniform
To see why attention didn't beat mean pooling, I measured on a single-seed model C for each target. The normalized entropy was .973 for surprise volatility, and .992 for raw volatility, with 1 being perfectly uniform. The average top weights were .042 for surprise, and .029 for raw volatility, compared with .017 for uniform. The most attended chunks only get about 3-4% of the weight.

Model C is effectively doing mean pooling. This explains why model C is roughly equivalent to model B in results. With no learnable content signal, there is no reason for attention to concentrate. For identity, every chunk carries the company's identifiers so averaging works just as well.

#### Case Studies of four misses

| Call | Paper's surprise label | A: Check-in 2 | B: Chunks + mean | C: Chunks + attention | Realized-vol surprise |
|---|---|---|---|---|---|
| META 2022 Q4 | +1.372 | −0.042 | −0.018 | −0.027 | +1.216 |
| LLY 2023 Q2 | +1.462 | +0.031 | −0.006 | −0.018 | +1.409 |
| T 2023 Q1 | +1.446 | −0.066 | −0.023 | −0.020 | +1.203 |
| DLR 2023 Q1 | −1.948 | −0.014 | +0.022 | +0.012 | +0.366 |

Every model predicts roughly zero for these four cases. For META, C's top weighted chunk is a discussion on expense cuts. This could have been what drove the volatility spike, but it only got .044 of the weight, so it was still averaged in with about 65 other chunks. Seeing the right content isn't enough if nothing in the 990 training calls taught the model what that content means for volatility.

For DLR, it was -1.95 on the paper's label, but +.37 on the realized volatility surprise. The target choice can flip the sign for a call, meaning that label choice matters.

I have a few ideas for why the models have failed. There could be too little data. 990 training calls to learn how content maps to volatility surprise might be insufficient. Additionally, the surprise target removes what the model actually learns because the company history is the only thing that the models found and surprise volatility removes that. Another potential failure is that MiniLM was trained on general web text, not finance specific. It might not be able to separate text in a way that the head can use. Lastly, targets may be noisy. Volatility also reacts to market and sector news unrelated to the call.

Additionally, there were several problems that came up during the process that I fixed before the final runs. The Q&A splitter needed four versions. The 200 word chunks would have exceeded MiniLM's 256 token limit so I re-chunked at 150 before embedding. Single seed tuning picked noise (best epochs were 0-2) so I switched to 3 seed selection. And the clustered bootstrap weakened a check in 2 claim.

### Takeaways and plan for the final submission

#### The Three Hypotheses
During this phase, all three hypotheses were tested, and none were supported. Hypothesis 1, chunking beating the single vector, was not supported for the surprise. However, the chunked representation is significantly better at encoding company-level information. 

For the hypothesis 2's claim that attention beats mean, this was not supported. The weights were near uniform so model C was roughly equal to model B.

For hypothesis 3's claim that signal lives in the Q&A, this was not supported. There was a non significant lean toward Q&A on the surprise, but there's no surprise signal for it to live in.

#### Main Claims from this Check In
So far, within this dataset, transcript text adds no detectable information about post-earnings volatility beyond the company's own history. This result holds across the single vector and chunked, the mean and attention pooling, both halves of the call, and two label definitions. 

The shuffle test shows the models learn which company is speaking. Any apparent text volatility signal needs to be checked against the company identity before it's believed. A text signal that only encodes which company is speaking is something a per-company average already captures, so it isn't useful as a predictor

The extension improved the representation. Chunking fixed truncation and does improve the representation (as the raw volatility positive control shows), but the improvement is all identity based. This means that the representation was not what was hiding a call specific signal.

#### Plan for Final Submission
Before the final submission, I will research and use a finance specific encoder. Potentially using FinBERT variants on Hugging Face which is trained on financial news. For the design, I will create two new models. Model D which combines finance encoder chunks with mean pooling, and model E which combines finance encoder chunks with attention. I will use the same splits, head, tuning procedure, and bootstrap. Comparing model D versus B will isolate the encoder. 

Another adjustment that I will make is time series related, which I am currently learning more about in DSAN 5100. Currently, everything rests on one split. Instead, I will use a walk-forward evaluation. I would have fold 1 train 2019-2020 and test on 2021, fold 2 would train on 2019-2021 and test on 2022, and fold 3 would train on 2019-2022 and test on 2023. This would increase the number of calls in my test set, so the bounds for my confidence intervals could get tighter. This would help me understand if results hold across different regimes. Validation in each fold would be the last year of each training window. This might cause some issues for fold 1 which will only train on 2019 and validate on 2020 (COVID year).

### AI Disclosure and References


#### AI Use
I used Claude in the following ways:
- Reseach: researching a proper encoder for this section (MiniLM), token counting (tiktoken), and ways to clean the encoding errors (ftfy), 
- Debugging code cells
- Adding comments to code cells to help reader know where everything is
- Helping me confirm that all of the numbers I wrote in the markdown accurately reflected my results
- How to properly implement the company clustered paired boostrap
- Failure Analysis: Shuffle test and attention entropy
- Ideas on how to accurately split the Q&A
- Check in 3.md: Markdown formatting of tables. Additionally, I used it to help fix sentences that ran too long or were hard to understand. Also used it to correct punctuation and spelling mistakes

#### References
Yu, D., Liu, Z., & He, H. (2025). Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts. *Findings of the Association for Computational Linguistics: ACL 2025.* https://aclanthology.org/2025.findings-acl.946/

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019.* https://arxiv.org/abs/1908.10084

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020).* https://arxiv.org/abs/2002.10957

Speer, R. ftfy: fixes text for you. https://github.com/rspeer/python-ftfy

OpenAI. tiktoken. https://github.com/openai/tiktoken

Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019).*