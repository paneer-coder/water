import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean text by removing special characters, numbers, and extra spaces"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """Lemmatize text and remove stop words"""
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(lemmatized)

# Example: Load your data
# df = pd.read_csv('your_data.csv')
# For demonstration, creating sample data
df = pd.DataFrame({
    'text': ['This is a sample text!', 'Another example with numbers 123.', 
             'Text preprocessing is important!!!'],
    'label': ['positive', 'negative', 'positive']
})

# Step 1: Text Cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Step 2: Lemmatization and Stop Words Removal
df['processed_text'] = df['cleaned_text'].apply(lemmatize_text)

# Step 3: Label Encoding
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# Step 4: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# Save outputs
df.to_csv('processed_data.csv', index=False)
np.save('tfidf_matrix.npy', tfidf_matrix.toarray())

# Save models for future use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Processing complete!")
print(f"Processed data shape: {df.shape}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print("\nSaved files:")
print("- processed_data.csv")
print("- tfidf_matrix.npy")
print("- label_encoder.pkl")
print("- tfidf_vectorizer.pkl")