import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Sample training data (in spaCy format)
TRAIN_DATA = [
    ("Apple Inc. is looking at buying U.K. startup for $1 billion", 
     {"entities": [(0, 10, "ORG"), (27, 31, "GPE"), (44, 55, "MONEY")]}),
    ("Elon Musk founded SpaceX in California in 2002", 
     {"entities": [(0, 9, "PERSON"), (18, 24, "ORG"), (28, 38, "GPE"), (42, 46, "DATE")]}),
    ("Google announced new AI products in San Francisco yesterday",
     {"entities": [(0, 6, "ORG"), (36, 49, "GPE"), (50, 59, "DATE")]}),
    ("Microsoft CEO Satya Nadella spoke at the conference",
     {"entities": [(0, 9, "ORG"), (14, 27, "PERSON")]}),
    ("The meeting is scheduled for Monday at 3 PM in New York",
     {"entities": [(29, 35, "DATE"), (39, 43, "TIME"), (47, 55, "GPE")]}),
]

TEST_DATA = [
    ("Amazon acquired Whole Foods for $13.7 billion in 2017",
     {"entities": [(0, 6, "ORG"), (16, 27, "ORG"), (32, 46, "MONEY"), (50, 54, "DATE")]}),
    ("Jeff Bezos visited Seattle last week",
     {"entities": [(0, 10, "PERSON"), (19, 26, "GPE"), (27, 36, "DATE")]}),
]


def train_ner_model(train_data, n_iter=30):
    """Train a custom NER model"""
    # Create blank English model
    nlp = spacy.blank("en")
    
    # Add NER pipeline component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels to NER
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Train the model
    optimizer = nlp.begin_training()
    
    print("Training the NER model...")
    for iteration in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Loss = {losses['ner']:.4f}")
    
    return nlp


def evaluate_ner(nlp, test_data):
    """Evaluate NER model and calculate metrics"""
    true_entities = []
    pred_entities = []
    
    for text, annotations in test_data:
        doc = nlp(text)
        
        # Get true entities
        true_ents = set()
        for start, end, label in annotations.get("entities"):
            true_ents.add((start, end, label))
        
        # Get predicted entities
        pred_ents = set()
        for ent in doc.ents:
            pred_ents.add((ent.start_char, ent.end_char, ent.label_))
        
        # Create labels for classification report
        all_positions = true_ents | pred_ents
        for pos in all_positions:
            true_label = pos[2] if pos in true_ents else "O"
            pred_label = pos[2] if pos in pred_ents else "O"
            true_entities.append(true_label)
            pred_entities.append(pred_label)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_entities, pred_entities, average='weighted', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(true_entities, pred_entities, zero_division=0)
    }


def predict_entities(nlp, text):
    """Extract entities from text"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Named Entity Recognition (NER) System")
    print("=" * 60)
    
    # Train the model
    nlp_model = train_ner_model(TRAIN_DATA, n_iter=30)
    
    print("\n" + "=" * 60)
    print("Evaluating on Test Data")
    print("=" * 60)
    
    # Evaluate the model
    metrics = evaluate_ner(nlp_model, TEST_DATA)
    
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['classification_report'])
    
    # Test on new examples
    print("\n" + "=" * 60)
    print("Testing on New Examples")
    print("=" * 60)
    
    test_sentences = [
        "Tesla CEO Elon Musk announced new products in Texas",
        "The conference in London starts on Friday at 10 AM",
        "Facebook changed its name to Meta in October 2021"
    ]
    
    for sentence in test_sentences:
        entities = predict_entities(nlp_model, sentence)
        print(f"\nText: {sentence}")
        print(f"Entities: {entities}")