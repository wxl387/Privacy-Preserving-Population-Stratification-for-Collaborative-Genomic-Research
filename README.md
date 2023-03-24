# Privacy-Preserving-Population-Stratification-for-Collaborative-Genomic-Research


 - `dataset` folder contains two datasets, one is from 1000genome, and the other is from openSNP.
 - `source_code` folder has the source code used to generate the model and perform the experiments in the Evaluation section. 
 `create_model.py` reduces the original high dimensional researchers' data and creates the ground truth by utilizing the k-means clustering algorithm. Then it splits the data into training and testing sets for privacy analysis. 
 `privacy_analysis.py` contains the code for the experiments included in the Evaluation section of the paper for various number of populations, number of PCA dimensions, and number of clusters. 
 - In order to reproduce the results, first `create_model.py` needs to be run on the dataset of your choice by editing the file directory in code, and then `privacy_analysis.py` to run the experiments. Please note that 1000genome has a larger size so it could be split into more testing sets than openSNP dataset when the size of the testing set is fixed. To avoid any errors, you may decrease the number of sampling from each group (you may also add `replace=True` to get sampling with replacement).
