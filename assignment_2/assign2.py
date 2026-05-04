import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import pandas as pd

# Sample data
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document"
]

# 1. Bag of Words - Count Occurrence
print("=== Bag of Words - Count Occurrence ===")
count_vectorizer = CountVectorizer()
bow_counts = count_vectorizer.fit_transform(documents)
bow_df = pd.DataFrame(bow_counts.toarray(), 
                      columns=count_vectorizer.get_feature_names_out())
print(bow_df)
print()

# 2. Bag of Words - Normalized Count Occurrence
print("=== Bag of Words - Normalized Count ===")
normalized_vectorizer = CountVectorizer(binary=False)
bow_normalized = normalized_vectorizer.fit_transform(documents)
# Normalize by dividing each row by its sum
bow_normalized_array = bow_normalized.toarray()
row_sums = bow_normalized_array.sum(axis=1, keepdims=True)
bow_normalized_array = bow_normalized_array / row_sums
bow_norm_df = pd.DataFrame(bow_normalized_array,
                           columns=normalized_vectorizer.get_feature_names_out())
print(bow_norm_df)
print()

# 3. TF-IDF
print("=== TF-IDF ===")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                        columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)
print()

# 4. Word2Vec Embeddings
print("=== Word2Vec Embeddings ===")
# Tokenize documents for Word2Vec
tokenized_docs = [doc.lower().split() for doc in documents]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_docs, 
                     vector_size=100, 
                     window=5, 
                     min_count=1, 
                     workers=4)

# Get vocabulary
print("Vocabulary:", list(w2v_model.wv.key_to_index.keys()))
print()

# Example: Get embedding for a word
word = "document"
if word in w2v_model.wv:
    print(f"Embedding for '{word}':")
    print(w2v_model.wv[word])
    print()

# Get document embeddings by averaging word vectors
def get_document_embedding(doc, model):
    words = doc.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.wv.vector_size)

print("Document embeddings (averaged word vectors):")
for i, doc in enumerate(documents):
    doc_embedding = get_document_embedding(doc, w2v_model)
    print(f"Document {i+1} embedding shape: {doc_embedding.shape}")