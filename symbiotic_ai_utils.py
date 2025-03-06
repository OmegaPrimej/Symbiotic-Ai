import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def process_input(input_text):
    # Tokenize the input text
    tokens = word_tokenize(input_text)
    
    # Remove stopwords and punctuation
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t.lower() not in stop_words]
    
    # Join the tokens back into a string
    input_text = ' '.join(tokens)
    return input_text

def generate_response(input_text, knowledge_base):
    # Vectorize the input text
    vectorizer = TfidfVectorizer()
    input_vector = vectorizer.fit_transform([input_text])
    
    # Vectorize the knowledge base
    knowledge_base_vectors = vectorizer.transform(knowledge_base)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(input_vector, knowledge_base_vectors)
    
    # Return the most similar knowledge base entry
    most_similar_index = similarities.argmax()
    return knowledge_base[most_similar_index]

Example usage:
knowledge_base = ["This is a sample knowledge base.", "Another piece of knowledge."]
input_text = "Hello, how are you?"
processed_input = process_input(input_text)
response = generate_response(processed_input, knowledge_base)
print("Response:", response)
