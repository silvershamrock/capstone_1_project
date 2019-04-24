# capstone_1_project
This repository contains the complete data and code for the Capstone 1 Project of the Springboard Data Science Career Track program. 
  
## About the project
Consumers of online news commonly share articles they find to be interesting or important with others on social media platforms. However, there is considerable variation in whether an individual decides to share any given article. What determines how popular a given article will be? Is it the subject of the article? The tone it is written in? Or perhaps simple structural characteristics like the length of the title and body, or how many links and images it contains? This project seeks to answer those questions by examining the relationship of such variables to social media shares using various analytical and machine learning methods. 

## The dataset
The dataset to be analyzed was published by previous researchers and donated to the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/machine-learning-databases/00332/). It covers approximately 40,000 articles published by the online media site Mashable over a period of two years. The response variable is a binary indicator of whether an article is “popular” or not (by specifying a threshold for number of shares across social media that balances popular/not popular classes in the training set). Independent variables to predict popularity (60), include the article, title, and average word length, how many links, images, and videos it contains, the popularity of linked articles and keywords within the article, the time elapsed between publication and retrieval of shares, the day of week of publication, what category it falls into (e.g. business, entertainment), topic classification as determined by Latent Dirichlet Allocation (LDA; a generative statistical model that measures term co-occurence), article/title sentiment polarity (negativity/positivity) and article/title subjectivity.

## Folder content
**data:** Contains the raw dataset used for analysis, as well as the output of scripts contained in code (e.g., graphs, model output, etc.)
**code:** Contains all of the code used to conduct analyses and produce graphs.
**reports:** Contains various reports describing the methods and findings of project analyses.
