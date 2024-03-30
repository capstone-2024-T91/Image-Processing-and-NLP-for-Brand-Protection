# Image-Processing-and-NLP-for-Brand-Protection


Malicious phishing websites change constantly and go through significant efforts to mimic legitimate websites. New phishing websites that impersonate companies are created every day, and always contain minor differences to evade automated detection methods. How could we use machine learning, natural language processing (NLP) and image recognition algorithms to automatically cluster and attribute malicious websites? The project will consist at developing a Machine Learning model for the early identification of Phishing sites.

## Repository Structure

### Classifiers

- `AdaBoostClassifier.joblib`: Serialized AdaBoost classifier model for making predictions.
- `DecisionTreeClassifier.joblib`: Serialized Decision Tree classifier model.
- `LogisticRegression.joblib`: Serialized Logistic Regression model.
- `RandomForestClassifier.joblib`: Serialized Random Forest classifier model.
- `SGDClassifier.joblib`: Serialized Stochastic Gradient Descent classifier model.
- `ml_detection.ipynb`: Jupyter notebook used for the training and validation of machine learning models.
- `ml_predictor.ipynb`: Jupyter notebook used for predicting labels on new data using the trained models.

### Neural_Networks

- `nn_phishing_model.h5`: Serialized neural network model saved in HDF5 format.
- `neural-network-detection.ipynb`: Jupyter notebook for training and validating neural network-based models.
- `nn_predictor.ipynb`: Jupyter notebook for making predictions with the neural network model.

### Tokenizers

- `tokenizer.pickle`: Serialized file containing the tokenizer for preprocessing text data before feeding it into the models.

### Vectorizers

- `AdaBoostClassifier_vectorizer.joblib`: Vectorizer paired with the AdaBoost model for data preprocessing.
- `DecisionTreeClassifier_vectorizer.joblib`: Vectorizer paired with the Decision Tree model.
- `LogisticRegression_vectorizer.joblib`: Vectorizer paired with the Logistic Regression model.
- `RandomForestClassifier_vectorizer.joblib`: Vectorizer paired with the Random Forest model.
- `SGDClassifier_vectorizer.joblib`: Vectorizer paired with the SGD classifier model.

### Other Files

- `Phishing_Email.csv`: The dataset containing email data, labeled for phishing detection.
- `README.md`: This document, which explains the project and repository structure.