# Methodology: Comparing Machine-Learning Models for Fake-News Detection

## 1. Research question
On the ISOT Fake and Real News dataset, how do Multinomial Naive Bayes and Logistic Regression compare when classifying English-language news articles using TF-IDF text features?

## 2. Dataset description
The project uses the ISOT Fake and Real News dataset, commonly distributed as two files: `Fake.csv` and `True.csv`. The expected files contain article fields such as `title`, `text`, `subject`, and `date`; labels are created from the source file (`FAKE` for `Fake.csv`, `REAL` for `True.csv`). Commonly reported dataset counts are approximately 23,481 fake and 21,417 real articles, but the actual counts in the downloaded copy will be reported by the notebook and may differ.

- **Input features:** Combined article title and body text (`title` + `text`).
- **Target:** Binary class label (`FAKE` or `REAL`).
- **Expected size:** About 44,900 rows before cleaning; actual cleaned size will be recorded after loading.
- **Source:** Use a legitimate dataset page/repository and record the exact download URL and access date in the final report.
- **Important limitations:** English-language content; source and time-period bias; labels inherited from dataset collection; duplicate or near-duplicate articles may exist; the dataset does not represent all countries, topics, languages, or present-day news.

## 3. Data cleaning plan
1. Load the two CSV files separately and create labels from their filenames.
2. Check columns, shape, missing values, label counts, and sample rows.
3. Keep the title and body text fields; treat a missing title or body as an empty string, then combine them.
4. Remove rows with empty combined text and exact duplicate texts.
5. Normalize whitespace and lowercase text. Remove URLs using a simple regular expression, but do not remove words or punctuation indiscriminately because they may carry information.
6. Inspect class balance and look for obvious label/source shortcuts (for example, terms such as `Reuters`, publisher names, or boilerplate). Document findings; do not claim that a high score proves factual verification.
7. Split data into training and test sets using an 80/20 stratified split with a fixed random seed.

## 4. Feature engineering
Use `TfidfVectorizer` with word unigrams and bigrams (`ngram_range=(1, 2)`) and a maximum vocabulary of 30,000 features. TF-IDF gives less weight to terms that occur in many documents and represents each article as a sparse numeric vector. Fit the vectorizer on training text only and transform the test text with the fitted vectorizer to avoid test-data leakage.

## 5. Models and rationale
- **Multinomial Naive Bayes:** A fast, commonly used baseline for text classification. It is simple and useful for comparison.
- **Logistic Regression:** A linear classifier that often works well with high-dimensional sparse TF-IDF features and provides a strong comparison with Naive Bayes.

Use the same train/test split and feature extraction settings for both models. Keep the random seed fixed for reproducibility.

## 6. Evaluation metrics
Report accuracy, precision, recall, F1-score, a classification report, and confusion matrices for both models. Since the dataset has two classes, report class-specific metrics and macro F1 as well as accuracy. F1 balances precision and recall and is useful if class counts are not perfectly balanced. Accuracy alone can hide uneven performance across classes.

## 7. Reproducibility and implementation
The experiment is implemented in `experiment.ipynb` using Python, pandas, matplotlib, and scikit-learn. Record the actual cleaned row count, class distribution, metrics, and observations from the executed notebook. Run all cells from top to bottom and preserve outputs before committing.

## 8. Ethical considerations and limitations
This is a benchmark classification experiment, not a fact-checking service. A model may learn source-specific vocabulary, formatting, or historical patterns instead of truthfulness. Results on this dataset may not generalize to new events, other languages, or different publishers. Future work could evaluate on a separately collected dataset, perform time-based and source-held-out testing, inspect influential words, and include human fact-checking or evidence retrieval.
