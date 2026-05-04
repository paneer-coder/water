import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import (
    WhitespaceTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "Hello! I'm working on NLP assignments. Natural Language Processing is amazing! #NLP @student"

print("Original Text:")
print(text)
print("\n" + "="*80 + "\n")

# 1. Whitespace Tokenization
print("1. WHITESPACE TOKENIZATION:")
ws_tokenizer = WhitespaceTokenizer()
ws_tokens = ws_tokenizer.tokenize(text)
print(ws_tokens)
print("\n" + "="*80 + "\n")

# 2. Punctuation-based Tokenization
print("2. PUNCTUATION-BASED TOKENIZATION:")
punct_tokenizer = WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(text)
print(punct_tokens)
print("\n" + "="*80 + "\n")

# 3. Treebank Tokenization
print("3. TREEBANK TOKENIZATION:")
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(text)
print(treebank_tokens)
print("\n" + "="*80 + "\n")

# 4. Tweet Tokenization
print("4. TWEET TOKENIZATION:")
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)
print(tweet_tokens)
print("\n" + "="*80 + "\n")

# 5. MWE (Multi-Word Expression) Tokenization
print("5. MWE TOKENIZATION:")
mwe_tokenizer = MWETokenizer([('Natural', 'Language'), ('Language', 'Processing')])
mwe_tokens = mwe_tokenizer.tokenize(punct_tokens)
print(mwe_tokens)
print("\n" + "="*80 + "\n")

# STEMMING
sample_words = ['working', 'works', 'worked', 'running', 'runs', 'easily', 'fairly']

# 6. Porter Stemmer
print("6. PORTER STEMMER:")
porter = PorterStemmer()
porter_stems = [porter.stem(word) for word in sample_words]
print(f"Original: {sample_words}")
print(f"Stemmed:  {porter_stems}")
print("\n" + "="*80 + "\n")

# 7. Snowball Stemmer
print("7. SNOWBALL STEMMER:")
snowball = SnowballStemmer('english')
snowball_stems = [snowball.stem(word) for word in sample_words]
print(f"Original: {sample_words}")
print(f"Stemmed:  {snowball_stems}")
print("\n" + "="*80 + "\n")

# LEMMATIZATION
print("8. LEMMATIZATION (WordNet):")
lemmatizer = WordNetLemmatizer()
sample_lemma_words = ['running', 'ran', 'runs', 'better', 'best', 'geese', 'feet']
lemmas = [lemmatizer.lemmatize(word, pos='v') if word in ['running', 'ran', 'runs'] 
          else lemmatizer.lemmatize(word) for word in sample_lemma_words]
print(f"Original:   {sample_lemma_words}")
print(f"Lemmatized: {lemmas}")