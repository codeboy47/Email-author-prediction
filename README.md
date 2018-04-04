# Email-author-prediction

Given the contents of an email predict whether the email's author is Chris(Label 0) or Sara(Label 1).

Five classifier algorithms are used for predictions:
1. Naive Bayes Classifier 
2. Support Vector Machines (SVM) Classifier
3. Decision Tree Classifier 
4. AdaBoost
5. Random forest 

| Algorithm | Training time(s) | Prediction time(s) | Accuracy |
| --- | --- |
| Naive Bayes Classifier | 1.745 | 0.188 | 0.973265073948 |
| Support Vector Machines Classifier | 129.289 | 12.482 | 0.990898748578 |
| Decision Tree Classifier | 16.955 | 0.032 | 0.976678043231 |
| AdaBoost Classifier | 96.273 | 0.342 | 0.956200227531 |
| Random Forest Classifier | 21.025 | 0.146 | 0.997155858931 |

Random Forest Classifier performs best in predicting email author as Chris or Sara with an accuracy of 0.997 or 99.7%.
