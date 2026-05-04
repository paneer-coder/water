"""
Assignment 5: Using WordNet to Identify Semantic Relationships
This program demonstrates the use of WordNet to identify:
- Synonymy (synonyms)
- Antonymy (antonyms)
- Hypernymy (more general terms/superordinate concepts)
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def get_synonyms(word, pos=None):
    """
    Get synonyms for a word using WordNet
    
    Args:
        word: The input word
        pos: Part of speech (optional): wn.NOUN, wn.VERB, wn.ADJ, wn.ADV
    
    Returns:
        set: A set of synonyms
    """
    synonyms = set()
    
    synsets = wn.synsets(word, pos=pos)
    
    for synset in synsets:
        for lemma in synset.lemmas():
            # Add the lemma name (synonym)
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    
    return synonyms


def get_antonyms(word, pos=None):
    """
    Get antonyms for a word using WordNet
    
    Args:
        word: The input word
        pos: Part of speech (optional)
    
    Returns:
        set: A set of antonyms
    """
    antonyms = set()
    
    synsets = wn.synsets(word, pos=pos)
    
    for synset in synsets:
        for lemma in synset.lemmas():
            # Check if the lemma has antonyms
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name().replace('_', ' '))
    
    return antonyms


def get_hypernyms(word, pos=None):
    """
    Get hypernyms (more general terms) for a word using WordNet
    
    Args:
        word: The input word
        pos: Part of speech (optional)
    
    Returns:
        set: A set of hypernyms
    """
    hypernyms = set()
    
    synsets = wn.synsets(word, pos=pos)
    
    for synset in synsets:
        # Get direct hypernyms
        for hypernym_synset in synset.hypernyms():
            for lemma in hypernym_synset.lemmas():
                hypernyms.add(lemma.name().replace('_', ' '))
    
    return hypernyms


def get_hyponyms(word, pos=None):
    """
    Get hyponyms (more specific terms) for a word using WordNet
    
    Args:
        word: The input word
        pos: Part of speech (optional)
    
    Returns:
        set: A set of hyponyms
    """
    hyponyms = set()
    
    synsets = wn.synsets(word, pos=pos)
    
    for synset in synsets:
        # Get direct hyponyms
        for hyponym_synset in synset.hyponyms():
            for lemma in hyponym_synset.lemmas():
                hyponyms.add(lemma.name().replace('_', ' '))
    
    return hyponyms


def get_word_definitions(word):
    """
    Get definitions for all synsets of a word
    
    Args:
        word: The input word
    
    Returns:
        list: List of (synset_name, definition) tuples
    """
    definitions = []
    synsets = wn.synsets(word)
    
    for synset in synsets:
        definitions.append((synset.name(), synset.definition()))
    
    return definitions


def analyze_word(word):
    """
    Comprehensive analysis of a single word using WordNet
    
    Args:
        word: The word to analyze
    
    Returns:
        dict: Dictionary containing all semantic relationships
    """
    print(f"\n{'='*60}")
    print(f"Analysis for word: '{word.upper()}'")
    print(f"{'='*60}")
    
    # Get definitions
    definitions = get_word_definitions(word)
    if definitions:
        print(f"\nDefinitions ({len(definitions)} senses):")
        for i, (synset_name, definition) in enumerate(definitions[:3], 1):
            print(f"  {i}. [{synset_name}] {definition}")
    
    # Get synonyms
    synonyms = get_synonyms(word)
    print(f"\nSynonyms ({len(synonyms)}): ", end="")
    if synonyms:
        print(", ".join(list(synonyms)[:10]))
    else:
        print("None found")
    
    # Get antonyms
    antonyms = get_antonyms(word)
    print(f"\nAntonyms ({len(antonyms)}): ", end="")
    if antonyms:
        print(", ".join(list(antonyms)))
    else:
        print("None found")
    
    # Get hypernyms
    hypernyms = get_hypernyms(word)
    print(f"\nHypernyms ({len(hypernyms)}): ", end="")
    if hypernyms:
        print(", ".join(list(hypernyms)[:10]))
    else:
        print("None found")
    
    # Get hyponyms
    hyponyms = get_hyponyms(word)
    print(f"\nHyponyms ({len(hyponyms)}): ", end="")
    if hyponyms:
        print(", ".join(list(hyponyms)[:10]))
    else:
        print("None found")
    
    return {
        'word': word,
        'definitions': definitions,
        'synonyms': synonyms,
        'antonyms': antonyms,
        'hypernyms': hypernyms,
        'hyponyms': hyponyms
    }


def analyze_text(text):
    """
    Analyze text data and extract semantic relationships for all meaningful words
    
    Args:
        text: Input text string
    
    Returns:
        dict: Dictionary containing analysis for each word
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens 
                     if word not in string.punctuation 
                     and word not in stop_words
                     and word.isalpha()]
    
    # Remove duplicates while preserving order
    unique_words = list(dict.fromkeys(filtered_words))
    
    print(f"\n{'='*60}")
    print(f"TEXT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nOriginal text: {text}")
    print(f"\nExtracted words for analysis: {', '.join(unique_words)}")
    
    results = {}
    
    for word in unique_words:
        results[word] = analyze_word(word)
    
    return results


def demonstrate_semantic_similarity():
    """
    Demonstrate semantic similarity using WordNet path similarity
    """
    print(f"\n{'='*60}")
    print(f"SEMANTIC SIMILARITY DEMONSTRATION")
    print(f"{'='*60}")
    
    word_pairs = [
        ('dog', 'cat'),
        ('car', 'automobile'),
        ('big', 'large'),
        ('good', 'bad'),
        ('walk', 'run')
    ]
    
    for word1, word2 in word_pairs:
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if synsets1 and synsets2:
            # Calculate path similarity between first synsets
            similarity = synsets1[0].path_similarity(synsets2[0])
            print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.3f}")
            print(f"  {word1}: {synsets1[0].definition()}")
            print(f"  {word2}: {synsets2[0].definition()}")


def main():
    """
    Main function to demonstrate WordNet semantic relationships
    """
    print("\n" + "="*60)
    print("WORDNET SEMANTIC RELATIONSHIPS ANALYZER")
    print("="*60)
    
    # Example 1: Analyze individual words
    print("\n\n--- EXAMPLE 1: Individual Word Analysis ---")
    words_to_analyze = ['happy', 'dog', 'computer', 'run', 'beautiful']
    
    for word in words_to_analyze:
        analyze_word(word)
    
    # Example 2: Analyze text
    print("\n\n--- EXAMPLE 2: Text Analysis ---")
    sample_text = "The quick brown dog runs happily through the beautiful garden."
    analyze_text(sample_text)
    
    # Example 3: Demonstrate semantic similarity
    print("\n\n--- EXAMPLE 3: Semantic Similarity ---")
    demonstrate_semantic_similarity()
    
    # Interactive mode
    print("\n\n--- INTERACTIVE MODE ---")
    while True:
        user_input = input("\nEnter a word to analyze (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using WordNet Semantic Relationships Analyzer!")
            break
        
        if user_input:
            analyze_word(user_input)
        else:
            print("Please enter a valid word.")


if __name__ == "__main__":
    main()
