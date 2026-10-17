# Fake News Detection Using TF-IDF and Machine Learning Classifiers

## Abstract

The rapid spread of misinformation online has increased the need for reliable methods of identifying fake news. This study compares two machine learning classifiers, Multinomial Naive Bayes and Logistic Regression, for binary fake news detection. The experiment uses the ISOT Fake and Real News dataset, containing labeled fake and real news articles. Text data are represented using Term Frequency–Inverse Document Frequency (TF-IDF), and both classifiers are evaluated using accuracy, precision, recall, and F1-score on the same test set. The test set contains 7,819 examples. The reported results show that Multinomial Naive Bayes achieved approximately 95% accuracy, while Logistic Regression achieved approximately 99%. These findings suggest that Logistic Regression performed better in this experimental setting. However, the results should be interpreted cautiously because performance on a particular dataset does not necessarily indicate effectiveness on new, unseen sources of news. Further evaluation on diverse and more recent datasets is needed.

1.Introduction-

Online news platforms and social media allow information to reach large audiences quickly. However, these platforms can also spread misleading or fabricated information. Fake news may influence public opinion, create confusion, and reduce trust in reliable information sources. Consequently, automatic fake news detection has become an important research topic in natural language processing and machine learning.
Previous research has explored different approaches to fake news detection. Shu et al. (2017) reviewed fake news detection on social media and discussed the importance of analyzing news content and social context. Wang (2017) introduced the LIAR dataset, which supports research into automated classification of statements with different truthfulness labels. Zhou and Zafarani (2020) surveyed fake news detection methods and highlighted challenges involving data, detection techniques, and the changing nature of misinformation.
Traditional machine learning methods provide a practical starting point for text classification. TF-IDF represents documents using numerical features based on the importance of words in a collection of documents. These features can then be used by classifiers to distinguish between categories.
This study compares Multinomial Naive Bayes and Logistic Regression using TF-IDF features to determine which model performs better on the selected fake and real news dataset.

### Research question

Which classifier, Multinomial Naive Bayes or Logistic Regression, achieves better performance in fake news detection when both use TF-IDF text features and are evaluated on the same test set?

## 2. Methodology

### 2.1 Dataset-

The experiment uses the ISOT Fake and Real News dataset, which contains news articles labeled as fake or real. The dataset files used in the project are `Fake.csv` and `True.csv`. The labels allow the task to be treated as a supervised binary classification problem.
The reported test set contains 7,819 examples: 3,580 fake news examples and 4,239 real news examples.

### 2.2 Text representation and preprocessing-

News articles are represented using Term Frequency–Inverse Document Frequency (TF-IDF). This technique assigns numerical weights to terms according to their frequency within a document and their distribution across the document collection. It converts text into features suitable for machine learning algorithms.
The text preparation and feature extraction steps follow the existing project notebook. Both classifiers use the same experimental dataset and TF-IDF representation to support a fair comparison.

### 2.3 Machine learning models-

Two classifiers are evaluated:

- Multinomial Naive Bayes: A probabilistic classifier commonly used for text classification. It estimates class probabilities from the observed features using the Naive Bayes assumption of conditional independence.
- **Logistic Regression:** A supervised classification algorithm that estimates class probabilities from input features and learns decision boundaries for distinguishing between categories.

Both models are trained using the prepared training data and evaluated on the held-out test data.

## 2.4 Evaluation metrics

The models are compared using four standard classification metrics:

-Accuracy: The proportion of all predictions that are correct.
-Precision: The proportion of predicted positive examples that are actually positive.
-Recall: The proportion of actual positive examples that are correctly identified.
-F1-score: The harmonic mean of precision and recall.

These metrics provide a broader view of performance than accuracy alone. The precise interpretation of class-specific precision, recall, and F1-score depends on the label treated as positive.

## 3. Results

## Table 1. Approximate performance comparison of the classifiers

| Evaluation metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 95% | 99% |
| Precision | 95% | 99% |
| Recall | 95% | 99% |
| F1-score | 95% | 99% |

*Note: Values are rounded approximations from the existing experiment summary. Replace them with the exact values from the notebook before final submission.*

Table 1 indicates that Logistic Regression achieved higher reported performance than Multinomial Naive Bayes across the four evaluation metrics. Its approximate accuracy was four percentage points higher.

### Table 2. Distribution of test-set examples

| News category | Number of examples | Percentage |
|      Fake     |        3,580       | 45.79% |
|      Real     |        4,239       | 54.21% |
|     *Total    |        7,819       | 100.00%|

Table 2 shows that the test set contains slightly more real news examples than fake news examples. Both categories have substantial representation, although the distribution is not perfectly balanced.
The results suggest that both classifiers can distinguish between fake and real news in this dataset, with Logistic Regression achieving the stronger reported performance.

## 4. Discussion

The experimental results favor Logistic Regression over Multinomial Naive Bayes for the selected dataset and TF-IDF representation. One possible explanation is that Logistic Regression can learn discriminative weights for individual text features, allowing it to separate the two classes effectively. Multinomial Naive Bayes uses a different probabilistic approach and relies on simplifying assumptions about feature independence.
However, the comparison does not establish that Logistic Regression is always superior for fake news detection. The outcome depends on the dataset, preprocessing, feature representation, parameter settings, and evaluation procedure.

### Limitations

This study has several limitations.

First, the experiment uses one dataset. Its results may not generalize to news from different countries, publishers, topics, or time periods.

Second, TF-IDF primarily represents textual patterns. It does not independently verify factual claims, examine the original evidence, or establish the credibility of a publisher.

Third, the performance metrics depend on the selected test set. Further experiments using additional datasets and different data splits would help assess the stability of the results.

Finally, the high reported accuracy of Logistic Regression should not be interpreted as proof that the model can reliably identify every form of misinformation. Dataset-specific patterns or similarities between training and test examples may influence performance.

Future work could evaluate additional classifiers, investigate feature selection and parameter tuning, test models on newer news articles, and examine confusion matrices and class-specific performance.

## 5. Conclusion

This study compared Multinomial Naive Bayes and Logistic Regression for fake news detection using TF-IDF features and the ISOT Fake and Real News dataset. The reported experiment showed that both classifiers performed well, while Logistic Regression achieved higher approximate accuracy, precision, recall, and F1-score. These findings support its use as a promising baseline for this particular classification task. Nevertheless, successful classification of a fixed dataset does not guarantee reliable detection of misinformation in real-world settings. Testing on diverse and recent news sources, reporting exact evaluation metrics, and investigating potential data leakage are important next steps toward developing more robust fake news detection systems.

## References

Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations Newsletter, 19*(1), 22–36. https://doi.org/10.1145/3137597.3137600

Wang, W. Y. (2017). “Liar, liar pants on fire”: A new benchmark dataset for fake news detection. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 422–426). Association for Computational Linguistics. https://doi.org/10.18653/v1/P17-2067

Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys, 53*(5), Article 109. https://doi.org/10.1145/3395046
