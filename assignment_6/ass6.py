"""
Assignment 6: Machine Translation System for English and Indian Languages
This program translates public information content between English and Indian languages
using Google Translate API through the googletrans library.
"""

from googletrans import Translator, LANGUAGES
import time
from typing import List, Dict, Tuple

# Initialize translator
translator = Translator()

# Indian Languages supported with their language codes
INDIAN_LANGUAGES = {
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'marathi': 'mr',
    'tamil': 'ta',
    'gujarati': 'gu',
    'urdu': 'ur',
    'kannada': 'kn',
    'malayalam': 'ml',
    'punjabi': 'pa',
    'odia': 'or',
    'assamese': 'as',
    'sanskrit': 'sa'
}


def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'hi') -> Dict:
    """
    Translate text from source language to target language
    
    Args:
        text: The text to translate
        source_lang: Source language code (default: 'auto' for auto-detection)
        target_lang: Target language code (default: 'hi' for Hindi)
    
    Returns:
        dict: Translation result containing original text, translated text, and metadata
    """
    try:
        # Perform translation
        result = translator.translate(text, src=source_lang, dest=target_lang)
        
        return {
            'success': True,
            'original_text': text,
            'translated_text': result.text,
            'source_language': result.src,
            'target_language': target_lang,
            'pronunciation': result.pronunciation if hasattr(result, 'pronunciation') else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_text': text
        }


def translate_batch(texts: List[str], source_lang: str = 'auto', target_lang: str = 'hi') -> List[Dict]:
    """
    Translate multiple texts in batch
    
    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        list: List of translation results
    """
    results = []
    
    for text in texts:
        result = translate_text(text, source_lang, target_lang)
        results.append(result)
        time.sleep(0.1)  # Small delay to avoid rate limiting
    
    return results


def bidirectional_translate(text: str, indian_lang: str = 'hi') -> Dict:
    """
    Translate text bidirectionally (English to Indian language and vice versa)
    
    Args:
        text: The text to translate
        indian_lang: Indian language code (default: 'hi')
    
    Returns:
        dict: Bidirectional translation results
    """
    # Detect source language
    detection = translator.detect(text)
    source_lang = detection.lang
    
    if source_lang == 'en':
        # English to Indian language
        result = translate_text(text, 'en', indian_lang)
        return {
            'direction': 'English to Indian Language',
            'result': result
        }
    else:
        # Indian language to English
        result = translate_text(text, indian_lang, 'en')
        return {
            'direction': 'Indian Language to English',
            'result': result
        }


def translate_public_announcement(announcement: str, target_languages: List[str]) -> Dict:
    """
    Translate a public announcement to multiple Indian languages
    
    Args:
        announcement: The public announcement text in English
        target_languages: List of target language codes
    
    Returns:
        dict: Translation results for all target languages
    """
    print(f"\n{'='*70}")
    print("PUBLIC ANNOUNCEMENT TRANSLATION")
    print(f"{'='*70}")
    print(f"\nOriginal Announcement (English):")
    print(f"{announcement}")
    print(f"\n{'='*70}")
    
    translations = {}
    
    for lang_code in target_languages:
        lang_name = [name for name, code in INDIAN_LANGUAGES.items() if code == lang_code]
        lang_name = lang_name[0].capitalize() if lang_name else lang_code
        
        result = translate_text(announcement, 'en', lang_code)
        
        if result['success']:
            translations[lang_name] = result['translated_text']
            print(f"\n{lang_name} ({lang_code}):")
            print(f"{result['translated_text']}")
        else:
            translations[lang_name] = f"Error: {result['error']}"
            print(f"\n{lang_name} ({lang_code}): Translation failed - {result['error']}")
        
        time.sleep(0.2)  # Delay to avoid rate limiting
    
    return translations


def translate_document(content: List[str], source_lang: str = 'en', target_lang: str = 'hi') -> List[Dict]:
    """
    Translate a document (multiple paragraphs/sentences)
    
    Args:
        content: List of paragraphs or sentences
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        list: Translation results for each paragraph
    """
    print(f"\n{'='*70}")
    print(f"DOCUMENT TRANSLATION: {source_lang.upper()} → {target_lang.upper()}")
    print(f"{'='*70}")
    
    results = []
    
    for i, paragraph in enumerate(content, 1):
        print(f"\n--- Paragraph {i} ---")
        print(f"Original: {paragraph}")
        
        result = translate_text(paragraph, source_lang, target_lang)
        
        if result['success']:
            print(f"Translated: {result['translated_text']}")
            results.append(result)
        else:
            print(f"Error: {result['error']}")
            results.append(result)
        
        time.sleep(0.2)
    
    return results


def detect_language(text: str) -> Dict:
    """
    Detect the language of input text
    
    Args:
        text: Input text
    
    Returns:
        dict: Detection result with language code and confidence
    """
    try:
        detection = translator.detect(text)
        lang_name = LANGUAGES.get(detection.lang, 'Unknown')
        
        return {
            'success': True,
            'language_code': detection.lang,
            'language_name': lang_name.capitalize(),
            'confidence': detection.confidence
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def display_menu():
    """Display the main menu"""
    print("\n" + "="*70)
    print("MACHINE TRANSLATION SYSTEM - ENGLISH ↔ INDIAN LANGUAGES")
    print("="*70)
    print("\nOptions:")
    print("1. Translate English to Indian Language")
    print("2. Translate Indian Language to English")
    print("3. Translate Public Announcement (Multiple Languages)")
    print("4. Translate Document/Multiple Paragraphs")
    print("5. Detect Language")
    print("6. View Supported Indian Languages")
    print("7. Run Demo Examples")
    print("8. Exit")
    print("="*70)


def display_indian_languages():
    """Display all supported Indian languages"""
    print(f"\n{'='*70}")
    print("SUPPORTED INDIAN LANGUAGES")
    print(f"{'='*70}")
    print(f"\n{'Language':<20} {'Code':<10}")
    print("-" * 70)
    
    for lang_name, lang_code in sorted(INDIAN_LANGUAGES.items()):
        print(f"{lang_name.capitalize():<20} {lang_code:<10}")
    
    print("="*70)


def run_demo_examples():
    """
    Run demonstration examples showing various translation scenarios
    """
    print("\n" + "="*70)
    print("DEMO EXAMPLES - MACHINE TRANSLATION SYSTEM")
    print("="*70)
    
    # Example 1: Simple English to Hindi translation
    print("\n--- Example 1: English to Hindi ---")
    text1 = "Hello, welcome to our public information system."
    result1 = translate_text(text1, 'en', 'hi')
    print(f"English: {result1['original_text']}")
    print(f"Hindi: {result1['translated_text']}")
    
    # Example 2: Hindi to English translation
    print("\n--- Example 2: Hindi to English ---")
    text2 = "नमस्ते, आपका स्वागत है।"
    result2 = translate_text(text2, 'hi', 'en')
    print(f"Hindi: {result2['original_text']}")
    print(f"English: {result2['translated_text']}")
    
    # Example 3: Public Announcement in multiple languages
    print("\n--- Example 3: Public Announcement ---")
    announcement = "Important Notice: All citizens are requested to follow safety guidelines."
    translate_public_announcement(announcement, ['hi', 'ta', 'te', 'bn'])
    
    # Example 4: Government Notice Translation
    print("\n--- Example 4: Government Notice Translation ---")
    notice = [
        "Government of India - Public Health Advisory",
        "Please maintain social distancing and wear masks in public places.",
        "For more information, visit our website or call the helpline."
    ]
    translate_document(notice, 'en', 'mr')
    
    # Example 5: Language Detection
    print("\n--- Example 5: Language Detection ---")
    test_texts = [
        "This is an English sentence.",
        "यह एक हिंदी वाक्य है।",
        "இது ஒரு தமிழ் வாக்கியம்.",
        "এটি একটি বাংলা বাক্য।"
    ]
    
    for text in test_texts:
        detection = detect_language(text)
        if detection['success']:
            print(f"\nText: {text}")
            print(f"Detected Language: {detection['language_name']} ({detection['language_code']})")
            print(f"Confidence: {detection['confidence']:.2f}")
    
    # Example 6: Educational Content Translation
    print("\n--- Example 6: Educational Content (English to Gujarati) ---")
    edu_content = "Education is the most powerful weapon which you can use to change the world."
    result6 = translate_text(edu_content, 'en', 'gu')
    print(f"English: {result6['original_text']}")
    print(f"Gujarati: {result6['translated_text']}")
    
    # Example 7: Emergency Alert
    print("\n--- Example 7: Emergency Alert (Multiple Languages) ---")
    emergency = "Emergency Alert: Heavy rainfall expected. Stay indoors and stay safe."
    translate_public_announcement(emergency, ['hi', 'te', 'kn'])


def interactive_mode():
    """
    Interactive mode for user input
    """
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            # English to Indian Language
            text = input("\nEnter English text: ").strip()
            if not text:
                print("Please enter valid text.")
                continue
            
            print("\nAvailable languages:")
            for i, (name, code) in enumerate(sorted(INDIAN_LANGUAGES.items()), 1):
                print(f"{i}. {name.capitalize()} ({code})")
            
            lang_choice = input("\nEnter language code (e.g., 'hi' for Hindi): ").strip().lower()
            
            if lang_choice not in INDIAN_LANGUAGES.values():
                print("Invalid language code!")
                continue
            
            result = translate_text(text, 'en', lang_choice)
            
            if result['success']:
                print(f"\nOriginal (English): {result['original_text']}")
                print(f"Translated: {result['translated_text']}")
            else:
                print(f"\nTranslation failed: {result['error']}")
        
        elif choice == '2':
            # Indian Language to English
            text = input("\nEnter text in Indian language: ").strip()
            if not text:
                print("Please enter valid text.")
                continue
            
            # Auto-detect or specify source language
            auto = input("Auto-detect language? (y/n): ").strip().lower()
            
            if auto == 'y':
                result = translate_text(text, 'auto', 'en')
            else:
                lang_code = input("Enter source language code: ").strip().lower()
                result = translate_text(text, lang_code, 'en')
            
            if result['success']:
                print(f"\nOriginal: {result['original_text']}")
                print(f"Detected/Source Language: {result['source_language']}")
                print(f"Translated (English): {result['translated_text']}")
            else:
                print(f"\nTranslation failed: {result['error']}")
        
        elif choice == '3':
            # Public Announcement
            announcement = input("\nEnter public announcement (English): ").strip()
            if not announcement:
                print("Please enter valid text.")
                continue
            
            print("\nSelect target languages (comma-separated codes, e.g., 'hi,ta,te'):")
            display_indian_languages()
            
            lang_input = input("\nEnter language codes: ").strip().lower()
            lang_codes = [code.strip() for code in lang_input.split(',')]
            
            # Validate language codes
            valid_codes = [code for code in lang_codes if code in INDIAN_LANGUAGES.values()]
            
            if not valid_codes:
                print("No valid language codes provided!")
                continue
            
            translate_public_announcement(announcement, valid_codes)
        
        elif choice == '4':
            # Document Translation
            print("\nEnter document content (enter 'END' on a new line to finish):")
            paragraphs = []
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                if line.strip():
                    paragraphs.append(line.strip())
            
            if not paragraphs:
                print("No content provided!")
                continue
            
            source = input("\nSource language code (or 'auto'): ").strip().lower()
            target = input("Target language code: ").strip().lower()
            
            if source != 'auto' and source not in INDIAN_LANGUAGES.values() and source != 'en':
                print("Invalid source language!")
                continue
            
            if target not in INDIAN_LANGUAGES.values() and target != 'en':
                print("Invalid target language!")
                continue
            
            translate_document(paragraphs, source, target)
        
        elif choice == '5':
            # Detect Language
            text = input("\nEnter text: ").strip()
            if not text:
                print("Please enter valid text.")
                continue
            
            detection = detect_language(text)
            
            if detection['success']:
                print(f"\nText: {text}")
                print(f"Detected Language: {detection['language_name']} ({detection['language_code']})")
                print(f"Confidence: {detection['confidence']:.2f}")
            else:
                print(f"\nDetection failed: {detection['error']}")
        
        elif choice == '6':
            # Display Supported Languages
            display_indian_languages()
        
        elif choice == '7':
            # Run Demo Examples
            run_demo_examples()
        
        elif choice == '8':
            # Exit
            print("\nThank you for using the Machine Translation System!")
            break
        
        else:
            print("\nInvalid choice! Please select 1-8.")
        
        input("\nPress Enter to continue...")


def main():
    """
    Main function
    """
    print("\n" + "="*70)
    print("MACHINE TRANSLATION SYSTEM")
    print("English ↔ Indian Languages")
    print("="*70)
    
    # Display supported languages
    print("\nSupported Indian Languages:")
    print(", ".join([name.capitalize() for name in INDIAN_LANGUAGES.keys()]))
    
    # Choose mode
    print("\n" + "="*70)
    print("Select Mode:")
    print("1. Demo Mode (Run examples)")
    print("2. Interactive Mode (User input)")
    print("="*70)
    
    mode = input("\nEnter choice (1-2): ").strip()
    
    if mode == '1':
        run_demo_examples()
        
        # Ask if user wants to continue to interactive mode
        continue_choice = input("\n\nDo you want to enter interactive mode? (y/n): ").strip().lower()
        if continue_choice == 'y':
            interactive_mode()
    elif mode == '2':
        interactive_mode()
    else:
        print("\nInvalid choice! Running demo mode by default...")
        run_demo_examples()


if __name__ == "__main__":
    main()
