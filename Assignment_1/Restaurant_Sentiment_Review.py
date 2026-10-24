#------------------------------- SECTION 1 : SETUP AND IMPORT ------------------------------------------------------------------


# 1. Standard Python libraries
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# 2. Data handling
import numpy as np
import pandas as pd

# 3. NLP and text preprocessing
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

# 4. Scikit-learn: text processing and machine learning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 5. Both machine learning algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# 6. Model evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 7. Reproducibility
SEED = 101
random.seed(SEED)
np.random.seed(SEED)

# 8. Download the NLTK resource required for lemmatization
nltk.download("wordnet", quiet=True)



#------------------------------- SECTION 2: STRUCTURED DATA SET CONSTRUCTION AND STORAGE ------------------------------------------------------------------


# ---------- QUESTION 1 : A dataset of formatted TEXT type data --------------------------------------------------------------------
# Structured like a CSV file -> each row = one text sample + its label

@dataclass
class RestaurantDatasetBuilder:
    """Creates and saves a restaurant review sentiment dataset."""

    output_filename: Path = Path("restaurant_reviews.csv")

    def _get_corpus_samples(self) -> Tuple[List[str], List[str]]:
        """Returns positive and negative restaurant reviews."""

        positive_reviews = [
            "The food was delicious and the service was excellent.",
            "Amazing atmosphere and very friendly staff.",
            "Fresh ingredients and generous portion sizes.",
            "The waiter was polite and attentive throughout dinner.",
            "Excellent food quality and reasonable prices.",
            "The restaurant was clean and beautifully decorated.",
            "Our meals arrived quickly and tasted wonderful.",
            "Great menu with plenty of vegetarian options.",
            "The desserts were delicious and beautifully presented.",
            "A wonderful dining experience. I will visit again.",
            "The staff handled our requests professionally.",
            "The soup was flavorful and served piping hot.",
            "Very comfortable seating and a relaxing atmosphere.",
            "The food was fresh, tasty, and worth the price.",
            "Excellent service from the moment we arrived.",
            "The pasta was perfectly cooked and delicious.",
            "A lovely restaurant with a welcoming environment.",
            "The portions were generous and the prices fair.",
            "Everything tasted fresh and was beautifully prepared.",
            "We had a fantastic dinner and will return soon.",
            "The pizza was crispy and full of flavor.",
            "The staff were helpful and made us feel welcome.",
            "The food arrived hot and was perfectly seasoned.",
            "A great place for a family dinner.",
            "The restaurant offered excellent value for money.",
            "The coffee was rich and the cake was delicious.",
            "The dining area was spacious and comfortable.",
            "Quick service and very tasty food.",
            "The chef prepared our meal exactly as requested.",
            "Everything was fresh and beautifully presented."
        ]

        negative_reviews = [
            "The food was cold and tasted terrible.",
            "The staff were rude and completely unhelpful.",
            "We waited an hour for our food to arrive.",
            "The restaurant was dirty and poorly maintained.",
            "Very expensive food with tiny portion sizes.",
            "The waiter ignored us throughout the meal.",
            "The soup was tasteless and served cold.",
            "Our order was incorrect and nobody fixed it.",
            "The meat was overcooked and extremely dry.",
            "The tables were sticky and the floor was dirty.",
            "Poor service and disappointing food quality.",
            "The restaurant was noisy and uncomfortable.",
            "The food arrived late and was barely warm.",
            "The menu was limited and overpriced.",
            "The staff seemed annoyed when we asked questions.",
            "The vegetables tasted old and unpleasant.",
            "We received the wrong dishes and waited again.",
            "The drinks were watered down and expensive.",
            "The bathroom was dirty and smelled unpleasant.",
            "A disappointing experience. I will not return.",
            "The pizza was soggy and lacked flavor.",
            "Nobody greeted us when we entered the restaurant.",
            "The food was greasy and difficult to eat.",
            "The restaurant was overcrowded and poorly organized.",
            "Our meals were overpriced and poorly prepared.",
            "The coffee was cold and tasted bitter.",
            "The chairs were uncomfortable and the tables cramped.",
            "Service was slow and the food was disappointing.",
            "The waiter brought the wrong drinks twice.",
            "The ingredients tasted stale and not fresh."
        ]

        return positive_reviews, negative_reviews

    def generate_and_save(self) -> pd.DataFrame:
        """Builds the dataset, shuffles it, and saves it as CSV."""

        positive, negative = self._get_corpus_samples()
        records = []

        # Add positive reviews with their labels
        for i, review in enumerate(positive, start=1):
            records.append({
                "review_id": f"REST-{i:03d}",
                "review_body": review,
                "sentiment": "positive"
            })

        # Add negative reviews with their labels
        for i, review in enumerate(
            negative, start=len(positive) + 1
        ):
            records.append({
                "review_id": f"REST-{i:03d}",
                "review_body": review,
                "sentiment": "negative"
            })

        # Convert records into a DataFrame
        df = pd.DataFrame(records)

        # Shuffle rows reproducibly
        df = df.sample(
            frac=1.0,
            random_state=SEED
        ).reset_index(drop=True)

        # Save the dataset to a CSV file
        df.to_csv(self.output_filename, index=False)

        print("Restaurant review dataset created successfully.")
        print(f"File saved as: {self.output_filename}")
        print(f"Dataset dimensions: {df.shape}")

        return df



# ---------- QUESTION 2: Build a PIPELINE using Data Processing and Feature Engineering -------------------------------------------------
#
# Pipeline steps:
#
# 1. Data Processing : lowercase text, expand contractions, remove
#    punctuation and numbers, tokenize, remove stopwords while preserving
#    negation words, and perform lemmatization.
#
# 2. Feature Engineering : convert processed text into numerical features
#    using Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer
#    with unigrams and bigrams).
#
# 3. Model Training : train a Naive Bayes classifier and a Logistic
#    Regression classifier using TF-IDF features.
#
# 4. Model Testing : test both models on the same unseen test data
#    using a 75:25 train-test split.
#
# 5. Model Evaluation : compare accuracy, confusion matrix, precision,
#    recall, and F1-score for both classifiers.
#
# 6. Sentiment Prediction : predict positive or negative sentiment
#    for new restaurant reviews using both trained models.


#------------------------------- SECTION 3: DATASET INTEGRATION AND VERIFICATION ------------------------------------------------------------------


def verify_dataset_structure(file_path: Path):
    """Loads the CSV and displays dataset diagnostics."""

    print("=" * 55)
    print("SECTION 3: DATASET STRUCTURE & INSPECTION")
    print("=" * 55)

    # Check whether the CSV file exists
    if not file_path.exists():
        print("Error: Dataset file not found!")
        print("Please generate the dataset first.")
        return

    # Read raw CSV lines directly from the file
    raw_lines = file_path.read_text(
        encoding="utf-8"
    ).splitlines()

    print("\n>>> CSV Header and First 5 Records:")
    for line in raw_lines[:6]:
        print("   ", line)

    print(f"\nTotal lines in CSV (including header): {len(raw_lines)}")

    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)

    print("\n>>> DataFrame Diagnostics:")
    print(f"Dimensions (Rows, Columns): {df.shape}")
    print(f"Missing Values: {df.isnull().sum().sum()}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nSentiment Distribution:")
    print(df["sentiment"].value_counts().to_string())

    print("\nDataset Information:")
    df.info()

    print("\nFirst 5 DataFrame Records:")
    print(df.head().to_string(index=False))

    print("=" * 55)





#------------------------------- SECTION 4: TEXT PROCESSING AND NORMALISATION ------------------------------------------------------------------



class TextNormalizer:
    """Cleans text using stopword removal, stemming and lemmatization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Preserve negation words because they affect sentiment
        self.negation_words = {
            "not", "no", "never", "nor", "neither", "cannot"
        }

        self.stop_words = set(ENGLISH_STOP_WORDS) - self.negation_words

    def preprocess_tokens(self, text: str):
        """Lowercases, expands contractions, tokenizes and removes stopwords."""

        text = str(text).lower()

        # Expand common contractions
        text = re.sub(r"\bwon't\b", "will not", text)
        text = re.sub(r"\bcan't\b", "cannot", text)
        text = re.sub(r"n't\b", " not", text)
        text = re.sub(r"'re\b", " are", text)
        text = re.sub(r"'ve\b", " have", text)
        text = re.sub(r"'ll\b", " will", text)
        text = re.sub(r"'d\b", " would", text)
        text = re.sub(r"'m\b", " am", text)

        # Remove punctuation and numbers
        text = re.sub(r"[^a-z\s]", " ", text)

        # Tokenize and remove stopwords
        tokens = [
            word for word in text.split()
            if word not in self.stop_words and len(word) > 1
        ]

        return tokens

    def transform(self, text: str) -> str:
        """Apply lemmatization for the machine learning pipeline."""

        tokens = self.preprocess_tokens(text)
        lemmas = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(lemmas)

    def stem_text(self, text: str) -> str:
        """Apply stemming to preprocessed text."""

        tokens = self.preprocess_tokens(text)
        stems = [self.stemmer.stem(word) for word in tokens]

        return " ".join(stems)





#------------------------------- SECTION 5: FEATURE ENGINEERING ------------------------------------------------------------------


def perform_feature_engineering(df: pd.DataFrame):
    """Demonstrates Bag-of-Words and TF-IDF feature extraction."""

    print("=" * 55)
    print("SECTION 5: FEATURE EXTRACTION & ENGINEERING")
    print("=" * 55)

    # Check that the processed text column exists
    if "processed_text" not in df.columns:
        raise ValueError(
            "Missing 'processed_text' column. "
            "Run text preprocessing first."
        )

    # 1. Bag-of-Words vectorization
    bow_vec = CountVectorizer()
    bow_matrix = bow_vec.fit_transform(df["processed_text"])

    print("\n>>> Bag-of-Words (CountVectorizer)")
    print("Feature matrix shape:", bow_matrix.shape)
    print(
        "Sample vocabulary:",
        list(bow_vec.get_feature_names_out()[:10])
    )

    # 2. TF-IDF vectorization using unigrams and bigrams
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vec.fit_transform(df["processed_text"])

    print("\n>>> TF-IDF (Unigrams + Bigrams)")
    print("Feature matrix shape:", tfidf_matrix.shape)
    print(
        "Sample features:",
        list(tfidf_vec.get_feature_names_out()[:10])
    )

    print("\nFeature extraction completed.")
    print("=" * 55)




#------------------------------- SECTION 6: MODEL TRAINING ------------------------------------------------------------------


class SklearnTextCleaner(BaseEstimator, TransformerMixin):
    """Wraps TextNormalizer for use inside a Scikit-learn pipeline."""

    def __init__(self):
        self.normalizer = TextNormalizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.normalizer.transform(text) for text in X]


def run_pipeline(df: pd.DataFrame):
    """Trains and compares Naive Bayes and Logistic Regression."""

    print("=" * 55)
    print("SECTION 6: PIPELINE ASSEMBLY & EVALUATION")
    print("=" * 55)

    # 1. Separate review text (X) and sentiment labels (y)
    X = df["review_body"]
    y = df["sentiment"]

    # 2. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")

    # 3. Define the Naive Bayes pipeline
    nb_pipeline = Pipeline([
        ("cleaner", SklearnTextCleaner()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])

    # 4. Define the Logistic Regression pipeline
    lr_pipeline = Pipeline([
        ("cleaner", SklearnTextCleaner()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(
            C=1.0,
            random_state=SEED,
            max_iter=1000
        ))
    ])

    # 5. Train and evaluate both models
    models = {
        "Naive Bayes": nb_pipeline,
        "Logistic Regression": lr_pipeline
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{'-' * 45}")
        print(f"MODEL: {model_name}")
        print("-" * 45)

        # Train the pipeline on training reviews
        model.fit(X_train, y_train)

        # Predict sentiment for unseen test reviews
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            "pipeline": model,
            "accuracy": accuracy
        }

        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("\nConfusion Matrix (negative, positive):")
        print(confusion_matrix(
            y_test,
            y_pred,
            labels=["negative", "positive"]
        ))

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=["negative", "positive"],
            zero_division=0
        ))

    # 6. Compare the two models
    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    for model_name, result in results.items():
        print(
            f"{model_name}: "
            f"{result['accuracy'] * 100:.2f}% accuracy"
        )

    # Select the model with the higher test accuracy
    best_model_name = max(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    print(f"\nHigher test accuracy: {best_model_name}")

    # Return both trained models and the selected model
    return results, best_model_name




#------------------------------- SECTION 7: INFERENCE ON UNSEEN SAMPLES------------------------------------------------------------------


def predict_new_reviews(results: dict):
    """Predicts sentiment for new reviews using both trained models."""

    print("=" * 60)
    print("SECTION 7: PREDICTION ON NEW RESTAURANT REVIEWS")
    print("=" * 60)

    # New reviews not included in our training dataset
    unseen_reviews = [
        "The food was delicious and the staff were wonderful.",
        "Terrible service and cold food. Very disappointing.",
        "The restaurant was okay, but the prices were too high.",
        "The waiter was rude and the food was not fresh.",
        "Amazing food, friendly staff, and a lovely atmosphere."
    ]

    # Test each trained model on the same new reviews
    for model_name, result in results.items():
        pipeline = result["pipeline"]

        print(f"\n>>> MODEL: {model_name}")
        print("-" * 60)

        predictions = pipeline.predict(unseen_reviews)
        probabilities = pipeline.predict_proba(unseen_reviews)

        # Find the probability associated with each predicted class
        classes = list(pipeline.named_steps["classifier"].classes_)

        for review, prediction, probability_row in zip(
            unseen_reviews, predictions, probabilities
        ):
            predicted_index = classes.index(prediction)
            confidence = probability_row[predicted_index]

            print(
                f"Sentiment: {prediction.upper():8s} | "
                f"Confidence: {confidence:.1%}"
            )
            print(f"Review: {review}\n")

    print("=" * 60)





# ------------------------------------------- SECTION 8 : MAIN EXECUTION -----------------------------------------------------------


if __name__ == "__main__":

    # QUESTION 1: Create the restaurant review dataset
    print("\nQUESTION 1: DATASET CREATION")

    builder = RestaurantDatasetBuilder()
    df_raw = builder.generate_and_save()

    # SECTION 3: Inspect the generated dataset
    verify_dataset_structure(builder.output_filename)

    # SECTION 4: Preprocess the review text
    print("\nQUESTION 2: TEXT PREPROCESSING")

    normalizer = TextNormalizer()

    # Demonstrate stemming and lemmatization
    sample_review = "The meals were delicious and the staff were serving wonderfully."

    print("\nOriginal review:")
    print(sample_review)

    print("\nAfter stemming:")
    print(normalizer.stem_text(sample_review))

    print("\nAfter lemmatization:")
    print(normalizer.transform(sample_review))

    df_raw["processed_text"] = df_raw["review_body"].apply(
        normalizer.transform
    )

    print("Text preprocessing completed.")
    print("\nOriginal review:")
    print(df_raw["review_body"].iloc[0])

    print("\nProcessed review:")
    print(df_raw["processed_text"].iloc[0])

    # SECTION 5: Demonstrate feature extraction
    print("\nQUESTION 3: FEATURE EXTRACTION")

    perform_feature_engineering(df_raw)

    # SECTION 6: Train and compare both models
    print("\nQUESTION 4: MODEL TRAINING AND EVALUATION")

    results, best_model_name = run_pipeline(df_raw)

    # SECTION 7: Predict sentiment for new reviews
    print("\nQUESTION 5: NEW REVIEW PREDICTIONS")

    predict_new_reviews(results)

    # Final summary
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Dataset saved to: {builder.output_filename}")
    print(f"Total reviews: {len(df_raw)}")
    print(f"Models trained: {len(results)}")
    print(f"Higher test accuracy: {best_model_name}")


# -------------------------------------------------------- CONCLUSION ----------------------------------------------------------------
#
# Q1: We created a formatted text dataset in CSV format containing
# restaurant reviews and their positive or negative sentiment labels.
#
# Q2: We developed an NLP pipeline involving text preprocessing,
# including lowercase conversion, punctuation and number removal,
# tokenization, stopword removal, stemming demonstration, and
# lemmatization. We extracted features using Bag of Words and TF-IDF
# with unigrams and bigrams. We trained and evaluated Naive Bayes
# and Logistic Regression classifiers and compared their performance.
# Finally, we used both models to predict the sentiment of new reviews.