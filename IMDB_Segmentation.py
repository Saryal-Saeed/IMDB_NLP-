import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



nltk.download('punkt')
nltk.download('stopwords')
Imdb_data=pd.read_csv('movies.csv')

description=Imdb_data['Description']



# Preprocess function to clean the description
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Apply the preprocessing function to the description column
Imdb_data['cleaned_description'] = Imdb_data['Description'].apply(preprocess_text)

genre_keywords = {
    'Action': ['fight', 'battle', 'war', 'explosion', 'hero', 'weapon'],
    'Romance': ['love', 'relationship', 'marriage', 'couple', 'romantic'],
    'Comedy': ['funny', 'humor', 'laugh', 'joke', 'comedy', 'hilarious'],
    'Drama': ['family', 'life', 'relationship', 'emotional', 'tragedy', 'conflict'],
    'Horror': ['scary', 'haunted', 'ghost', 'kill', 'death', 'fear', 'horror'],
    'Thriller': ['suspense', 'mystery', 'crime', 'chase', 'detective', 'danger'],
    'Science Fiction': ['alien', 'future', 'space', 'robot', 'technology', 'sci-fi', 'science'],
    'Fantasy': ['magic', 'kingdom', 'dragon', 'wizard', 'mythical', 'fantasy'],
}


# Function to classify genre based on keywords
def classify_genre(tokens):
    genre_matches = []
    
    for genre, keywords in genre_keywords.items():
        # Check if any of the genre keywords appear in the tokens
        if any(keyword in tokens for keyword in keywords):
            genre_matches.append(genre)
    
    # If one or more genres match, return them as a string
    if genre_matches:
        return ', '.join(genre_matches)
    else:
        return 'Unknown'  # If no genre matches, label it as 'Unknown'

# Apply the genre classification function to the cleaned descriptions
Imdb_data['Predicted_Genre'] = Imdb_data['cleaned_description'].apply(classify_genre)


print(Imdb_data['Predicted_Genre'])