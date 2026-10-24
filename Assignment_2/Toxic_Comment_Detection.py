# Questions

# 1. Implement a complete Word2Vec + LSTM text-classification pipeline.
# 2. Fine-tune a pretrained Transformer-DistilBERT-for text classification.


#---------------------------------------------INSTALL AND IMPORT REQUIRED LIBRARIES------------------------------------------------

# INSTALL LIBRARIES REQUIRED

# Run this in your VS Code terminal once:-
# python -m pip install torch datasets gensim numpy pandas scikit-learn matplotlib seaborn transformers accelerate


# IMPORT LIBRARIES

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU only: no CUDA
DEVICE = torch.device("cpu")
print("Training device:", DEVICE)


#-------------------------------------------QUESTION 1:  WORD2VEC + LSTM CLASSIFICATION PIPELINE---------------------------------------


#-------------------------------------------STEP 1: LOAD A SMALL TOXIC COMMENTS DATASET -----------------------------------------------

def get_sample(split, count, seed):
    stream = load_dataset(
        "google/civil_comments",
        split=split,
        streaming=True
    )
    stream = stream.shuffle(
        seed=seed,
        buffer_size=10_000
    )
    rows = list(stream.take(count))
    return HFDataset.from_list(rows)

train_data = get_sample("train", 5000, SEED)
valid_data = get_sample("validation", 1000, SEED + 1)
test_data = get_sample("test", 1000, SEED + 2)

print("Training examples:", len(train_data))
print("Validation examples:", len(valid_data))
print("Testing examples:", len(test_data))

print("\nSample comment:")
print(train_data[0]["text"])
print("Toxicity score:", train_data[0]["toxicity"])



#-------------------------------------------STEP 2: CLEAN THE TEXT ------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " EMAIL ",
        text
    )

    # Keep letters, digits, spaces and apostrophes
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


TOXICITY_THRESHOLD = 0.5

def prepare_data(data):
    texts = [clean_text(text) for text in data["text"]]

    labels = np.array(
        [
            1 if score >= TOXICITY_THRESHOLD else 0
            for score in data["toxicity"]
        ],
        dtype=np.float32
    )

    return texts, labels


train_texts, train_labels = prepare_data(train_data)
valid_texts, valid_labels = prepare_data(valid_data)
test_texts, test_labels = prepare_data(test_data)

# Check the number of examples in each class
print("Training labels:", np.unique(train_labels, return_counts=True))
print("Validation labels:", np.unique(valid_labels, return_counts=True))
print("Testing labels:", np.unique(test_labels, return_counts=True))


# Tokenize comments
train_sentences = [text.split() for text in train_texts]
valid_sentences = [text.split() for text in valid_texts]
test_sentences = [text.split() for text in test_texts]

print("Original comment:")
print(train_data[0]["text"])

print("\nCleaned comment:")
print(train_texts[0])

print("\nTraining label counts:")
print(pd.Series(train_labels).value_counts().sort_index())



#-------------------------------------------STEP 3: TRAIN A Word2Vec MODEL ----------------------------------------------------------

EMBEDDING_SIZE = 50

word2vec_model = Word2Vec(
    sentences=train_sentences,
    vector_size=EMBEDDING_SIZE,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=5,
    seed=SEED
)

print(
    "Word2Vec vocabulary size:",
    len(word2vec_model.wv.index_to_key)
)

if "comment" in word2vec_model.wv:
    print(
        "Vector shape:",
        word2vec_model.wv["comment"].shape
    )

    print("\nWords similar to 'comment':")
    for word, similarity in word2vec_model.wv.most_similar(
        "comment", topn=5
    ):
        print(f"{word:15s} {similarity:.3f}")



#-------------------------------------------STEP 4: CREATE WORD IDs AND THE EMBEDDING MATRIX -----------------------------------------

word_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in word2vec_model.wv.index_to_key:
    word_to_id[word] = len(word_to_id)

vocab_size = len(word_to_id)
print("Vocabulary size:", vocab_size)

embedding_matrix = np.zeros(
    (vocab_size, EMBEDDING_SIZE),
    dtype=np.float32
)

embedding_matrix[1] = np.random.normal(
    0, 0.05, EMBEDDING_SIZE
)

for word, word_id in word_to_id.items():
    if word in word2vec_model.wv:
        embedding_matrix[word_id] = word2vec_model.wv[word]

print("Embedding matrix shape:", embedding_matrix.shape)


#-------------------------------------------STEP 5: CONVERT COMMENTS INTO FIXED-LENGTH SEQUENCES --------------------------------------

MAX_WORDS = 100

def make_sequence(words):
    words = words[:MAX_WORDS]

    if not words:
        words = ["<UNK>"]

    sequence = [
        word_to_id.get(word, 1)
        for word in words
    ]

    actual_length = len(sequence)

    sequence += [0] * (MAX_WORDS - actual_length)

    return sequence, actual_length


train_encoded = [make_sequence(x) for x in train_sentences]
valid_encoded = [make_sequence(x) for x in valid_sentences]
test_encoded = [make_sequence(x) for x in test_sentences]

train_sequences = [x[0] for x in train_encoded]
valid_sequences = [x[0] for x in valid_encoded]
test_sequences = [x[0] for x in test_encoded]

train_lengths = [x[1] for x in train_encoded]
valid_lengths = [x[1] for x in valid_encoded]
test_lengths = [x[1] for x in test_encoded]

print("Training sequences:", len(train_sequences))
print("Sequence length:", len(train_sequences[0]))
print("First 15 word IDs:", train_sequences[0][:15])



#-------------------------------------------STEP 6: CREATE PyTorch DATALOADERS -------------------------------------------------------

class ToxicCommentDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = torch.tensor(
            sequences, dtype=torch.long
        )
        self.lengths = torch.tensor(
            lengths, dtype=torch.long
        )
        self.labels = torch.tensor(
            labels, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.lengths[index],
            self.labels[index]
        )


BATCH_SIZE = 32

train_loader = DataLoader(
    ToxicCommentDataset(
        train_sequences, train_lengths, train_labels
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    ToxicCommentDataset(
        valid_sequences, valid_lengths, valid_labels
    ),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    ToxicCommentDataset(
        test_sequences, test_lengths, test_labels
    ),
    batch_size=BATCH_SIZE
)

example_sequences, example_lengths, example_labels = next(
    iter(train_loader)
)

print("Batch shape:", example_sequences.shape)
print("Length shape:", example_lengths.shape)
print("Label shape:", example_labels.shape)


#-------------------------------------------STEP 7: BUILD THE BiLSTM MODEL -------------------------------------------------------------

#The bidirectional LSTM processes each comment from left to right and right to left. 
#It combines both representations to produce a single toxicity score.

class ToxicityBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0
        )

        # Initialize embeddings with Word2Vec vectors
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, sequence, lengths):
        embedded = self.embedding(sequence)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Combine final forward and backward hidden states
        final_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )

        final_hidden = self.dropout(final_hidden)

        return self.output_layer(final_hidden).squeeze(1)


lstm_model = ToxicityBiLSTM(
    vocab_size,
    EMBEDDING_SIZE,
    embedding_matrix
).to(DEVICE)

print(lstm_model)

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in lstm_model.parameters()
        if p.requires_grad
    )
)



#-------------------------------------------STEP 8: TRAIN  Word2Vec + LSTM MODEL ----------------------------------------------------

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=0.001
)

EPOCHS = 5

training_losses = []
validation_accuracies = []


def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, lengths, labels in data_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences, lengths)
            predictions = (
                torch.sigmoid(outputs) >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


for epoch in range(1, EPOCHS + 1):
    lstm_model.train()
    total_loss = 0

    for sequences, lengths, labels in train_loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = lstm_model(sequences, lengths)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    validation_accuracy = calculate_accuracy(
        lstm_model, valid_loader
    )

    training_losses.append(average_loss)
    validation_accuracies.append(validation_accuracy)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation accuracy: {validation_accuracy:.4f}"
    )



#-------------------------------------------STEP 9: SHOW THE TRAINING GRAPH ---------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(training_losses, marker="o", color="red")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(validation_accuracies, marker="o", color="green")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()



#-------------------------------------------STEP 10: TEST THE Word2Vec + BiLSTM MODEL --------------------------------------------------

lstm_model.eval()

test_predictions = []
test_probabilities = []

with torch.no_grad():
    for sequences, lengths, labels in test_loader:
        sequences = sequences.to(DEVICE)

        outputs = lstm_model(sequences, lengths)

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        test_probabilities.extend(probabilities)
        test_predictions.extend(predictions)

test_predictions = np.array(test_predictions, dtype=int)
test_probabilities = np.array(test_probabilities)

lstm_accuracy = accuracy_score(
    test_labels.astype(int), test_predictions
)

lstm_f1 = f1_score(
    test_labels.astype(int),
    test_predictions,
    zero_division=0
)

print("WORD2VEC + BiLSTM TEST RESULTS")
print("Accuracy:", round(lstm_accuracy, 4))
print("F1-score:", round(lstm_f1, 4))

print("\nDetailed classification report:")
print(
    classification_report(
        test_labels.astype(int),
        test_predictions,
        labels=[0, 1],
        target_names=["Non-toxic", "Toxic"],
        digits=4,
        zero_division=0
    )
)

#-------------------------------------------STEP 11: CONFUSION MATRIX FOR LSTM ------------------------------------------------------

lstm_cm = confusion_matrix(
    test_labels.astype(int),
    test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    lstm_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-toxic", "Toxic"],
    yticklabels=["Non-toxic", "Toxic"]
)

plt.title("Word2Vec + BiLSTM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()


#-------------------------------------------STEP 12: PREDICT A NEW REVIEW WITH LSTM ----------------------------------------------------

def predict_with_lstm(comment):
    cleaned_text = clean_text(comment)
    words = cleaned_text.split()

    sequence, length = make_sequence(words)

    sequence_tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(DEVICE)

    lengths_tensor = torch.tensor(
        [length],
        dtype=torch.long
    )

    lstm_model.eval()

    with torch.no_grad():
        output = lstm_model(
            sequence_tensor,
            lengths_tensor
        )

        probability = torch.sigmoid(output).item()

    predicted_class = 1 if probability >= 0.5 else 0

    predicted_name = (
        "Toxic" if predicted_class == 1
        else "Non-toxic"
    )

    return predicted_name, probability


new_comment = "Thank you for sharing your helpful ideas."

prediction, probability = predict_with_lstm(new_comment)

print("Comment:", new_comment)
print("Predicted class:", prediction)
print("Toxicity probability:", round(probability, 4))


#-------------------------------------------STEP 13: SAVE THE TRAINED Word2Vec +  BiLSTM MODEL  -------------------------------------------------------------

os.makedirs("saved_models", exist_ok=True)

# Save trained LSTM weights
torch.save(
    lstm_model.state_dict(),
    "saved_models/toxic_bilstm.pt"
)

# Save trained Word2Vec model
word2vec_model.save(
    "saved_models/word2vec.model"
)

# Save vocabulary mapping
with open(
    "saved_models/word_to_id.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(word_to_id, f)

print("Word2Vec + BiLSTM model saved successfully!")
print("Location: saved_models/")




#-------------------------------------------QUESTION 2: FINE-TUNE A PRETRAINED TRANSFORMER—DistilBERT FOR TEXT CLASSIFICATION---------------------------------------


#-------------------------------------------STEP 1: LOAD DistilBERT TOKENIZER AND MODEL ------------------------------------------------

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)

print("Model:", MODEL_NAME)
print("Tokenizer vocabulary size:", len(bert_tokenizer))
print("Maximum token length:", MAX_LENGTH)




#-------------------------------------------STEP 2: TOKENIZE THE SAME DATASET ----------------------------------------------------------

# Use the same google/civil_comments dataset from Question 1.
# Convert toxicity scores into binary labels:
# 0 = Non-toxic
# 1 = Toxic

def tokenize_comments(data):
    texts = list(data["text"])

    labels = [
        1 if score >= TOXICITY_THRESHOLD else 0
        for score in data["toxicity"]
    ]

    bert_dataset = HFDataset.from_dict({
        "text": texts,
        "labels": labels
    })

    # Tokenize comments using the DistilBERT tokenizer
    def tokenize_batch(batch):
        return bert_tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    bert_dataset = bert_dataset.map(
        tokenize_batch,
        batched=True
    )

    # Remove original text after tokenization
    bert_dataset = bert_dataset.remove_columns(["text"])

    # Convert token IDs, attention masks and labels into PyTorch tensors
    bert_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return bert_dataset


# Tokenize the same dataset splits used in Question 1
bert_train = tokenize_comments(train_data)
bert_valid = tokenize_comments(valid_data)
bert_test = tokenize_comments(test_data)

# Reuse the original test texts and labels from Question 1
bert_test_texts = list(test_data["text"])
bert_test_labels = test_labels.astype(np.int64)

print("Training dataset:", len(bert_train))
print("Validation dataset:", len(bert_valid))
print("Testing dataset:", len(bert_test))

print("\nExample input IDs:")
print(bert_train[0]["input_ids"][:20])

print("\nExample attention mask:")
print(bert_train[0]["attention_mask"][:20])

print("\nExample label:")
print(bert_train[0]["labels"])



#-------------------------------------------STEP 3: DEFINE EVALUATION METRICS -----------------------------------------------------------

def compute_metrics(evaluation_result):
    logits = evaluation_result.predictions
    labels = evaluation_result.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }



#-------------------------------------------STEP 4: FINE-TUNE DistilBERT MODEL -----------------------------------------------------------

training_args = TrainingArguments(
    output_dir="distilbert_output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
    seed=SEED,
    use_cpu=True
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=bert_train,
    eval_dataset=bert_valid,
    compute_metrics=compute_metrics
)

trainer.train()



#-------------------------------------------STEP 5: EVALUATE DistilBERT ON THE VALIDATION SET -------------------------------------------

validation_results = trainer.evaluate(
    eval_dataset=bert_valid
)

print("\nDistilBERT VALIDATION RESULTS")

for metric_name, metric_value in validation_results.items():
    if isinstance(metric_value, (int, float)):
        print(f"{metric_name}: {metric_value:.4f}")



#-------------------------------------------STEP 6: TEST DistilBERT ON THE SAME TEST DATASET --------------------------------------------

test_output = trainer.predict(bert_test)

test_logits = test_output.predictions

if isinstance(test_logits, tuple):
    test_logits = test_logits[0]

bert_test_predictions = np.argmax(
    test_logits,
    axis=1
)

bert_test_probabilities = torch.softmax(
    torch.tensor(test_logits),
    dim=1
).numpy()

bert_accuracy = accuracy_score(
    bert_test_labels,
    bert_test_predictions
)

bert_f1 = f1_score(
    bert_test_labels,
    bert_test_predictions,
    zero_division=0
)

print("\nDistilBERT TEST RESULTS")
print("Accuracy:", round(bert_accuracy, 4))
print("F1-score:", round(bert_f1, 4))

print("\nDetailed classification report:")

print(
    classification_report(
        bert_test_labels,
        bert_test_predictions,
        labels=[0, 1],
        target_names=["Non-toxic", "Toxic"],
        digits=4,
        zero_division=0
    )
)



#-------------------------------------------STEP 7: CONFUSION MATRIX FOR DistilBERT -----------------------------------------------------

bert_cm = confusion_matrix(
    bert_test_labels,
    bert_test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    bert_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Non-toxic", "Toxic"],
    yticklabels=["Non-toxic", "Toxic"]
)

plt.title("DistilBERT Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()



#-------------------------------------------STEP 8: PREDICT A NEW COMMENT WITH DistilBERT -----------------------------------------------

def predict_with_distilbert(comment):
    encoded = bert_tokenizer(
        comment,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0].cpu().numpy()

    predicted_class = int(np.argmax(probabilities))

    predicted_name = (
        "Toxic" if predicted_class == 1
        else "Non-toxic"
    )

    return predicted_name, float(probabilities[1])


# Use an example from the SAME test dataset
new_comment = test_data[0]["text"]

prediction, toxicity_probability = predict_with_distilbert(
    new_comment
)

print("\nComment:", new_comment)

actual_label = (
    "Toxic" if bert_test_labels[0] == 1
    else "Non-toxic"
)

print("Actual label:", actual_label)
print("Predicted class:", prediction)
print("Toxicity probability:", round(toxicity_probability, 4))



#-------------------------------------------STEP 9: SAVE THE FINE-TUNED DistilBERT MODEL -----------------------------------------------

os.makedirs("saved_models", exist_ok=True)

DISTILBERT_SAVE_PATH = "saved_models/distilbert_toxicity_model"

bert_model.save_pretrained(DISTILBERT_SAVE_PATH)
bert_tokenizer.save_pretrained(DISTILBERT_SAVE_PATH)

print("\nFine-tuned DistilBERT model saved successfully!")
print("Location:", DISTILBERT_SAVE_PATH)



#-------------------------------------------STEP 10: FINAL OUTPUT AND COMPARISON --------------------------------------------------------

final_results = pd.DataFrame({
    "Model": [
        "Word2Vec + BiLSTM",
        "Fine-tuned DistilBERT"
    ],
    "Type": [
        "Recurrent Neural Network",
        "Pretrained Transformer"
    ],
    "Test Accuracy": [
        f"{lstm_accuracy:.4f}",
        f"{bert_accuracy:.4f}"
    ],
    "Test F1-score": [
        f"{lstm_f1:.4f}",
        f"{bert_f1:.4f}"
    ]
})

print("\nFINAL COMPARISON")
print(final_results.to_string(index=False))


#---------------------------------------------------CONCLUSION--------------------------------------------------------------------

#In Question 1, Word2Vec learned word vectors from the Civil Comments training dataset, and those vectors were passed to a BiLSTM for
#toxicity classification.

#In Question 2, DistilBERT was already pretrained on a large text corpus. Fine-tuning adjusted it foR the toxic/non-toxic comment 
#classification task using the same Civil Comments dataset.

#DistilBERT may achieve better accuracy because its self-attention mechanism helps it understand the context of words in a comment.

#Word2Vec + BiLSTM is easier to understand and can require fewer computational resources, while DistilBERT may provide
#better classification performance. The actual performance of both models depends on the experimental results.