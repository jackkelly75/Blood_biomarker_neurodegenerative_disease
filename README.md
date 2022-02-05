# Thesis_FeatureSelection_ML_code

Code for feature selection and classification approaches used in Chapter 6 of thesis.  
`Optimised_Hyperparameter_models` contains the code run and the results for optimsation and running of feature selection and classification
<br/><br/>
Data is imported in R. It is then processed and analysed in python.

<br/>

Cleaned and processed data is available at Mendeley data DOI: 10.17632/88myp7kgrr.1

<br/>
Data citation:  

Kelly, Jack (2022), “Identifying blood biomarkers of neurodegenerative diseases using machine learning”, Mendeley Data, V1, doi: 10.17632/88myp7kgrr.1

<br/>

Abstract:  

This analysis aims to identify blood-based biomarkers for AD and PD by applying machine learning approaches. Multiple feature selection methods are used including a knowledge-based feature pool incorporating genes identified in previous analysis of AD and PD data. These consequently derived feature sets are used with various classification algorithms to identify the best approach and biomarkers for AD and PD datasets individually. Additionally, deep learning algorithms are applied to reduce the feature dimensionality of data to evaluate their potential in future work on gene expression biomarkers.  

To AD the best random forest model trained with 159 genes identified using VSSRFE with logistic regression (ROC AUC = 0.886) while to PD, the best random forest model is identified with all genes included in the dataset (ROC AUC = 0.743). CNN with a softmax classifier performs consistently well across both AD and PD datasets, suggesting its good potential in gene expression biomarker detection. Using knowledge-based feature pools did not inherently improve classification performance over using all genes in the dataset, suggesting that when looking for biomarkers of ND a genes importance in pathophysiology of the disease does not translate to biomarker potential.
