
# Toxic Comment Classification using Word2Vec + BiLSTM and DistilBERT

This project uses NLP and Deep Learning to classify online comments as toxic or non-toxic.

## Features
- Text preprocessing and cleaning
- Word2Vec word embeddings
- BiLSTM text classification
- Fine-tuning pretrained DistilBERT
- Model evaluation using accuracy and F1-score
- Confusion matrix and performance comparison
- Toxicity prediction on new comments
- Saving trained models

## Technologies Used
- Python
- PyTorch
- Gensim
- Hugging Face Datasets and Transformers
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn

## Dataset
The project uses a subset of the Google Civil Comments dataset (`google/civil_comments`).

- Training: 5,000 comments
- Validation: 1,000 comments
- Testing: 1,000 comments
- `0` = Non-toxic
- `1` = Toxic (toxicity score >= 0.5)

Both models use the same dataset splits and labels.

## Output
- Training loss and validation accuracy graphs
- Classification reports for both models
- Confusion matrices
- Toxicity predictions and probabilities for new comments
- Comparison table of test accuracy and F1-score
- Saved models in the `saved_models/` directory

## Conclusion
Word2Vec + BiLSTM uses learned word embeddings and a recurrent neural network for toxicity classification.
DistilBERT uses pretrained contextual representations and is fine-tuned on the same dataset. Both models are compared using test accuracy and F1-score.

