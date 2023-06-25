
# Enhancing Machine Learning Predictive Models for Anomaly Detection in API Security
This project focused on enhancing the accuracy of machine learning models for predicting anomalous behavior in API security. The primary objective was to predict the 'score' extracted from the 'behavior_type' variable in a high-dimensional dataset. As a key researcher, I implemented Principal Component Analysis (PCA) to reduce the dimensions of the original dataset and create a new feature set with features that are relatively independent of each other. I developed four methodologies to generate the value of our feature 'score' and designed the architecture of the deep neural network used for predicting the continuous value of score. The machine learning models used for two-class prediction were Support Vector Machines (SVMs) and Random Forest. The project resulted in a more accurate prediction of the 'score' variable, demonstrating the effectiveness of PCA and different methodologies in enhancing the accuracy of machine learning models for anomaly detection in API security.

This code extracts the source data and runs a deep neural network to predict the scores based on the algorithm discussed in the paper.

Box #1 imports the relevant libraries.

Box #2 imports the data, expands the data using the variable 'expand' and 'iteration_count', and runs the 2 component PCA over the data.

Box #3 generates the additional score column using user-defined distributions (chosen from 4 distributions)

Box #4 generates the test and training data

Box #5 runs the n-componenet PCA analysis and redefines the test and training data based on the results from PCA

Box #6 generates the deep neural network and compiles it

Box #7 Generates the accuracy scores.

Box #8 genertes the distribution of data in each feature set and scatter plot between few variables.

Box #9 plots the accuracy as the dataset increases 
