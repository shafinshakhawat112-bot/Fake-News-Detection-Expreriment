# Fake News Detection Using TF-IDF and Machine Learning Classifiers

## Abstract

The spread of fake news has become a significant challenge in the digital information environment. Machine learning techniques can help identify misleading information by analysing patterns in textual content. This study investigates the performance of two machine learning classifiers, Multinomial Naive Bayes and Logistic Regression, for fake news detection using Term Frequency–Inverse Document Frequency (TF-IDF) features.

The experiment uses the ISOT Fake and Real News Dataset. After removing empty articles and duplicate content within each label, 39,095 articles were retained. The models were evaluated using accuracy, macro precision, macro recall, and macro F1-score. Logistic Regression achieved an accuracy of 98.77%, outperforming Multinomial Naive Bayes, which achieved 95.38%. The findings demonstrate the effectiveness of TF-IDF-based machine learning approaches on this dataset, while also highlighting the need to consider dataset-specific patterns when interpreting performance.

## 1. Introduction

Online news platforms and social media have made information easier to access and share. However, they have also increased the circulation of fake news and misleading information. The rapid spread of inaccurate content can influence public opinion and create confusion.

Manual fact-checking requires time and human effort. Therefore, automated fake news detection has become an important research area in natural language processing and machine learning.

This project explores whether machine learning classifiers can distinguish between fake and real news articles using their textual content.

**Research Question:** How effectively can Multinomial Naive Bayes and Logistic Regression classify fake and real news articles using TF-IDF features?

## 2. Methodology

### 2.1 Dataset

The experiment uses the ISOT Fake and Real News Dataset, consisting of two files:

* `Fake.csv`: Fake news articles.
* `True.csv`: Real news articles.

The original dataset contained 44,898 articles. After removing empty text entries and duplicate content within each label, the dataset contained 39,095 articles:

| Category  | Number of Articles |
| --------- | -----------------: |
| Fake News |             17,900 |
| Real News |             21,195 |
| **Total** |         **39,095** |

The dataset contains the columns `title`, `text`, `subject`, `date`, and `label`. The experiment uses the combined title and article text as model input. Subject and date are not used as direct features.

### 2.2 Data Preprocessing

The following preprocessing operations were applied:

1. Missing titles and text values were replaced with empty strings.
2. Article titles and main text were combined into a single text field.
3. Text was converted to lowercase.
4. URLs were removed.
5. Whitespace was normalised.
6. Articles with empty text were removed.
7. Duplicate content was removed within each label.

These steps prepared the textual data for feature extraction and model training.

### 2.3 Feature Extraction: TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert news articles into numerical feature vectors.

The vectoriser was configured with the following parameters:

| Parameter                  | Value   |
| -------------------------- | ------- |
| Maximum features           | 30,000  |
| N-gram range               | (1, 2)  |
| Minimum document frequency | 2       |
| Accent handling            | Unicode |

Unigrams and bigrams were included to capture individual words and two-word combinations.

The TF-IDF vectoriser was fitted on the training data and then used to transform the test data. This ensured that the vocabulary and inverse document frequency values were learned from the training set rather than the test set.

### 2.4 Machine Learning Models

Two supervised machine learning algorithms were implemented.

**Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification. It estimates the probability of a class based on the frequency of words and other text features.

**Logistic Regression**

Logistic Regression is a supervised classification algorithm that estimates the probability of an input belonging to a particular class. It is commonly applied to high-dimensional text classification problems.

Both models were trained using the TF-IDF feature representations.

### 2.5 Train-Test Data

The dataset was divided into training and testing sets.

| Dataset      | Number of Articles |
| ------------ | -----------------: |
| Training set |             31,276 |
| Testing set  |              7,819 |
| **Total**    |         **39,095** |

The test set contained 3,580 fake news articles and 4,239 real news articles.

### 2.6 Evaluation Metrics

The models were evaluated using four metrics:

* **Accuracy:** Measures the proportion of correctly classified articles.
* **Macro Precision:** Calculates precision for each class and takes their unweighted average.
* **Macro Recall:** Calculates recall for each class and takes their unweighted average.
* **Macro F1-score:** Calculates the F1-score for each class and takes their unweighted average.

Macro-averaged metrics provide a view of performance across both classes.

## 3. Results

The two classifiers produced the following results on the test dataset:

| Evaluation Metric | Multinomial Naive Bayes | Logistic Regression |
| ----------------- | ----------------------: | ------------------: |
| Accuracy          |                  95.38% |              98.77% |
| Macro Precision   |                  95.38% |              98.80% |
| Macro Recall      |                  95.32% |              98.73% |
| Macro F1-score    |                  95.35% |              98.76% |

Logistic Regression achieved the highest performance across all four evaluation metrics.

Its accuracy was 98.77%, compared with 95.38% for Multinomial Naive Bayes. This represents an improvement of 3.39 percentage points in accuracy.

## 4. Discussion

The experimental results indicate that both classifiers performed effectively on the ISOT dataset. However, Logistic Regression consistently outperformed Multinomial Naive Bayes.

One possible explanation is that Logistic Regression can learn more flexible decision boundaries from high-dimensional TF-IDF features. This may help it capture useful word patterns and combinations in the news articles.

The results also demonstrate that TF-IDF can provide useful textual representations for traditional machine learning classifiers.

However, high test accuracy does not necessarily mean that a model can reliably identify fake news in every real-world situation. Models may learn dataset-specific writing styles, source patterns, or formatting signals rather than general indicators of misinformation.

An additional investigation of frequent terms revealed notable differences between the two labels. For example, the term "Reuters" appeared in 1.23% of fake articles and 99.82% of real articles. Similarly, "Breaking" appeared in 6.59% of fake articles and 1.80% of real articles, while "Read more" appeared in 9.53% of fake articles and 0.04% of real articles.

These patterns suggest that the classifiers may benefit from source-related or formatting-related signals in the dataset. Such signals could contribute to high classification performance without necessarily representing reliable evidence of misinformation.

## 5. Limitations

This experiment has several limitations:

1. **Dataset dependency:** The models were evaluated using one dataset. Their performance may differ on news articles from other sources or platforms.
2. **Dataset-specific patterns:** The classifiers may learn writing styles, source names, or formatting patterns associated with the dataset labels.
3. **Duplicate investigation:** Duplicate content was removed within each label, but cross-label duplicate content was not separately investigated.
4. **Limited model selection:** Only Multinomial Naive Bayes and Logistic Regression were evaluated.
5. **No external validation:** The models were not tested on an independent external news dataset.

Future research could address these limitations by using additional datasets, investigating cross-label duplicates, testing more advanced models, and evaluating performance on unseen news sources.

## 6. Conclusion

This project investigated fake news detection using TF-IDF feature extraction and two machine learning classifiers: Multinomial Naive Bayes and Logistic Regression.

Both models achieved strong results on the ISOT Fake and Real News Dataset. Logistic Regression performed better, achieving 98.77% accuracy and a macro F1-score of 98.76%, while Multinomial Naive Bayes achieved 95.38% accuracy and a macro F1-score of 95.35%.

The findings suggest that traditional machine learning algorithms can be effective for text classification. Nevertheless, the results should be interpreted carefully because dataset-specific patterns may influence model performance.

Future work should focus on improving generalisation, evaluating models on independent datasets, and developing methods that identify misleading information rather than simply recognising writing styles or source patterns.

## 7. References

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. *ACM SIGKDD Explorations Newsletter, 19*(1), 22–36. doi:10.1145/3137597.3137600

2. Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, 422–426. doi:10.18653/v1/P17-2067

3. Zhou, X., & Zafarani, R. (2020). A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities. *ACM Computing Surveys, 53*(5), Article 109. doi:10.1145/3395046
