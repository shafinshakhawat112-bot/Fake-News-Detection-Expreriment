# Literature Review: Fake-News Detection Using Machine Learning

## 1. Shu et al. (2017) — Fake News Detection on Social Media: A Data Mining Perspective
Shu and colleagues review how researchers define, characterize, and detect fake news. They explain that article text alone can be insufficient and that social context may help, which means a text classifier should not be treated as a definitive fact-checker.

Source: https://doi.org/10.1145/3137597.3137600

## 2. Ahmed, Traore, and Saad (2017) — Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques
This paper investigates text-classification methods for detecting online fake news, including n-gram features and traditional machine-learning algorithms. It supports using a transparent text-processing baseline, while also reminding us that a model's performance depends on the dataset and the way its examples were collected.

Source: https://doi.org/10.1007/978-3-319-69155-8_9

## 3. Shu et al. (2018) — FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media
The authors introduce FakeNewsNet as a resource that brings together news content and contextual information for fake-news research. The repository documentation also explains that the complete data cannot simply be redistributed because of platform privacy and publisher copyright restrictions, illustrating why dataset provenance and limitations should be documented.

Source: https://arxiv.org/abs/1809.01286
Dataset repository: https://github.com/KaiDMML/FakeNewsNet

## 4. scikit-learn documentation — TfidfVectorizer
The scikit-learn documentation describes how TF-IDF converts documents into numerical features that give less weight to terms common across many documents. This provides a practical and reproducible way to represent news text before training traditional classifiers; the vectorizer must be fitted on training data only to avoid leaking information from the test set.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

## 5. Ahmed, Traore, and Saad (2018) — Detecting Opinion Spams and Fake News Using Text Classification
This paper examines text-classification approaches for opinion spam and fake-news detection. It is relevant to this project because it shows how conventional text features and supervised learning can be used as a baseline, but the findings should not be assumed to transfer perfectly to current news from different countries or publishers.

Source: https://doi.org/10.1002/spy2.9

---

## Connection to this project
Based on these sources, this project compares Multinomial Naive Bayes and Logistic Regression using TF-IDF features. The goal is to compare their performance on one labelled dataset, not to claim that the resulting model can verify every real-world claim. Dataset bias, source-specific writing patterns, duplicate articles, and changes over time are important limitations.
