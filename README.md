# TCAV for Explaining Text Classifiers 

This repository replicates results in following ACL2022 publication: 

Nejadgholi, I. Fraser, K. C., Kiritchenko, S. (2022). Improving Generalizability in Implicitly Abusive Language Detection with Concept Activation Vectors. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics. https://arxiv.org/pdf/2204.02261.pdf

## Data

1.  Raw data:
`annotated_tweets_w_text.csv` includes the raw data used in the paper. This data contains information about [COVID-HATE (CH)](http://claws.cc.gatech.edu/covid/).

It is hard to locate or reconstruct the EA data used in the paper. However, it is a subset of the original data. The original data is available in the following links: [East-Asian Prejudice (EA)](https://zenodo.org/record/3816667#.YUJPkJ1KiUk)


2. Indexes of implicit/explicit abuse:
[In question] `CH_Anti_Asian_hate_implicit_indexes.csv` and `CH_Anti_Asianhate_explicit_indexes.csv` include indexes of implicitly and explicitly hateful samples in the _Anti-Asian Hate_ class of the _CH_ dataset, respectively. These indexes correspond to indexes of the `annotated_tweets_w_text.csv` file from the original dataset.  

`EA_dev_hostile_implicit_ids.csv` and `EA_dev_hostile_explicit_ids.csv` include tweet ids of implicitly and explicitly hostile samples of the _EA-dev_ set. 

3. Random stopwords tweets
Because the authors didn't include this txt file in the repository, I can only try to find some alternative. I found one data that might be a good substitute for this data set (http://help.sentiment140.com/for-students/). Then I included the text of 2000 random tweets, separated by double newlines.

The data is a CSV with emoticons removed. Data file format has 6 fields:
- 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- 1 - the id of the tweet (2087)
- 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
- 4 - the user that tweeted (robotickilldozr)
- 5 - the text of the tweet (Lyx is cool)


## Software

### Python modules:
 
`Roberta_model_data.py`: Roberta model and functions to compute gradients and logits of a roberta-based classifier

`TCAV.py`: fuctions to claculate sensitivities of a trained classifier to a human-defined concept (TCAV scores described in Section 4 of the paper) 

`DoE.py`: functions to calcualte the Degree of Explicitness (DoE scores described in Sections 5 and 6 of teh paper)

### Notebooks:

`wikidata.ipynb`: This notebook demonstrates how to construct the training, development and test sets of the _Toxicity_ classifier.

`roberta_train`: This notebook shows how to train a binary roberta-based classifier with [Wiki]. 

`EACHdata.ipynb`: This notebook shows how to create the following text files. Each file include the text of tweets separated by double newlines.     
 &nbsp;&nbsp;&nbsp;&nbsp; `data/EA_dev_implicit.txt`, `data/EA_dev_explicit.txt`, `data/CH_implicit.txt` and `data/CH_explicit.txt`

These notebooks illusterate how to use the above functionalities. In all of the notebooks, the _Toxicity_ classifier refers to a roberta-based binary classifier trained with the [Wiki](https://github.com/IsarNejad/cross_dataset_toxicity) dataset. 

`TCAV_Example.ipynb`: This notebook shows how to calculate the sensitivity of a trained classifier to a human-defined concept (similar to the results in Table 5 of the paper.  

[Pending implementation]`DoE_example.ipynb`: This colab notebook calcuates the Degree of Explicitness (DoE scores introduced in section 5 of the paper). 