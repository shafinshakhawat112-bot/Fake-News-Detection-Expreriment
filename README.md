# Fake News Detection Using TF-IDF and Machine Learning Classifiers

## Project Overview

This project investigates the use of machine learning techniques to classify news articles as either **Fake** or **Real**.

It compares two supervised machine learning algorithms:

* Multinomial Naive Bayes
* Logistic Regression

The models use TF-IDF (Term Frequency–Inverse Document Frequency) to convert news articles into numerical features.

The experiment uses the ISOT Fake and Real News Dataset.

## Research Objective

The main objective is to evaluate and compare the performance of Multinomial Naive Bayes and Logistic Regression in detecting fake news based on textual information.

**Research Question:** How effectively can these two machine learning classifiers distinguish between fake and real news using TF-IDF features?

## Dataset

**Dataset:** ISOT Fake and Real News Dataset

| Category  |   Articles |
| --------- | ---------: |
| Fake News |     17,900 |
| Real News |     21,195 |
| **Total** | **39,095** |

The original dataset contained 44,898 articles. After preprocessing and removing duplicate content within each label, 39,095 articles remained.

## Technologies and Tools

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorisation
* Multinomial Naive Bayes
* Logistic Regression
* Matplotlib

## Methodology

The experiment follows these main steps:

1. Load the fake and real news datasets.
2. Clean and preprocess the article text.
3. Combine article titles and main text.
4. Convert text into numerical features using TF-IDF.
5. Train Multinomial Naive Bayes and Logistic Regression.
6. Evaluate both models using the test dataset.
7. Compare their classification performance.

### TF-IDF Configuration

| Parameter                  | Value  |
| -------------------------- | ------ |
| Maximum features           | 30,000 |
| N-gram range               | (1, 2) |
| Minimum document frequency | 2      |

The TF-IDF vectoriser was fitted on the training data and used to transform the test data.

## Experimental Results

The models were evaluated using accuracy, macro precision, macro recall, and macro F1-score.

| Metric          | Multinomial Naive Bayes | Logistic Regression |
| --------------- | ----------------------: | ------------------: |
| Accuracy        |                  95.38% |              98.77% |
| Macro Precision |                  95.38% |              98.80% |
| Macro Recall    |                  95.32% |              98.73% |
| Macro F1-score  |                  95.35% |              98.76% |

### Key Finding

Logistic Regression achieved the best overall performance, with **98.77% accuracy**, outperforming Multinomial Naive Bayes by 3.39 percentage points.

These results apply to the ISOT dataset used in this experiment and do not guarantee equivalent performance on independent real-world news sources.

## Project Files

| File                                    | Description                |
| --------------------------------------- | -------------------------- |
| `experiment.ipynb`                      | Main experimental notebook |
| `paper.md`                              | Research paper             |
| `Fake_News_Detection_Presentation.pptx` | Project presentation       |
| `literature_review.md`                  | Literature review          |
| `methodology.md`                        | Research methodology       |
| `requirements.txt`                      | Python dependencies        |

## Limitations

* The experiment uses one dataset.
* Dataset-specific writing styles and source patterns may influence model performance.
* Cross-label duplicate content was not separately investigated.
* The models were not evaluated on an independent external dataset.

## Future Work

Future improvements may include:

* Testing additional machine learning algorithms.
* Evaluating models on independent datasets.
* Investigating cross-label duplicate content.
* Exploring deep learning and transformer-based approaches.
* Improving the models' ability to generalise to unseen news sources.

## Conclusion

The experiment demonstrates that traditional machine learning algorithms can perform effectively in fake news classification using TF-IDF features.

Logistic Regression achieved better results than Multinomial Naive Bayes across all four evaluation metrics.

However, further testing on independent datasets is necessary to assess real-world reliability.

## Author

**Shafin Shakhawat**

GitHub: [shafinshakhawat112-bot](https://github.com/shafinshakhawat112-bot)
