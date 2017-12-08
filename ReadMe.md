# DeepPPI - Refactoring for Keras 2

This code is just a refactoring of the original code associated with the article [DeepPPI: Boosting Prediction of Protein-Protein Interactions with Deep Neural Networks.](https://www.ncbi.nlm.nih.gov/pubmed/28514151) by Du et al, J Chem Inf Model. 2017 Jun 26;57(6):1499-1510. doi: 10.1021/acs.jcim.7b00028. Epub 2017 May 26. The sole purpose is to make the code run with Keras >= 2.0. All the credit for the analysis goes to the authors. The original code and data can be retrieved from [this website](http://ailab.ahu.edu.cn:8087/DeepPPI/index.html).

##DeepPPI

Theano/Keras implementation of a deep learning network designed for Protein Protein Interaction Prediction.

###DeepPPI uses the following dependencies:
* python 2.7
* numpy 1.11.1
* scipy 0.18.1
* HDF5 and h5py 
* scikit-learn 0.18
* Theano 0.8.2
* keras 1.2.0

###Guiding principles:
**Test_DeepPPI_Sep_Con_On_Five_Subset:**
DeepPPI compare with different architectures. The prediction result of our method on the five S. cerevisiae subset.

**DeepPPI_Compare_With_Traditional_Algorithm_With_Optimizing:**
We compare DeepPPI with several traditional prediction approaches, including Nearest Neighbors,
SVM, Decision Tree, Random Forest,AdaBoost, Naive Bayes, Quadratic Discriminant Analysis(QDA).

**DeepPPI_Compare_Existing_Methods:**
Performance on the S. cerevisiae core subset, H. pylori Dataset and Human Dataset

**DeepPPI_Corss_Vaild_On_Three_Dataset:**
Five fold cross Vaild on the S. cerevisiae core subset, H. pylori Dataset and Human Dataset

**DeepPPI_Predict_Five_Across_Species:**
we use all 34,514 samples of the S. cerevisiae dataset as the training set
and other species datasets (E. coli, C. elegans, H. sapiens, H. pylori and M.musculus) as test sets.

**DeepPPI_Corss_Vaild_On_Human/Yeast_Gold_Dataset:**
DeepPPI uses ten fold cross validation on human/yeast gold dataset.

**DeepPPI_Corss_Vaild_On_Human/Yeast_Silver_Dataset:**
DeepPPI uses ten fold cross validation on human/yeast silver dataset.

**DeepPPI_Predict_In_All_Human/Yeast_Dataset:**
DeepPPI training in Gold and Silver Datasets and Testing in All Interactions
Dataset

**Traditional_Algorithm_Optimizing:**
We use grid search to find the optimal parameters of traditional algorithms

###Runing

run these file from command line. 

*For example:*
```
>python DeepPPI_Predict_Five_Across_Species.pyc
output:
E.coli Acc:92.19
C.elegans Acc:94.84
H.sapiens Acc:93.77
H.pylori Acc:93.66
M.musculus Acc:91.37
```
