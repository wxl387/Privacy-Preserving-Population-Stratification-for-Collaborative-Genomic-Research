# Privacy-Preserving-Population-Stratification-for-Collaborative-Genomic-Research


 - `dataset` folder contains two datasets, one is from 1000genome, and the other is from openSNP.
 - `source_code` folder has two python files. `create_model.py` reduces the original high dimensional data and create ground truth by kmeans clustering. Then it splits training and several testing sets for privacy analysis. `privacy_analysis.py` runs for evaluations for various Number of populations, Number of PCA dimensions, and Number of clusters. 
 - First run `create_model.py` on the dataset of your choice by editing the file directory in code, then run `privacy_analysis.py`. Please notice, 1000genome has a larger size so it could be split into more testing sets than openSNP when the size of testing set is fixed. To avoid error, you could decrease the number of sampling from each group to obtain more number of testing sets. You could also add `replace=True` to the sampling.




