
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Setup
stopwords = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Clean text: remove stopwords, punctuation, lowercase
def clean_text(text, stop_words=False):
    text = str(text).lower()
    if stop_words:
        text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

# Lemmatize text
def lemmatizing(text):
    return [wn.lemmatize(word) for word in text.split()]

# Full preprocessing pipeline
def preprocess_text_data(df, stop_words=False):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Clean and lemmatize
    df['cleaned'] = df['comment_text'].apply(clean_text, stop_words=stop_words)
    df['lemmatized'] = df['cleaned'].apply(lemmatizing)

    # # Compute if comment is toxic in any way
    # df['is_toxic'] = df[label_cols].sum(axis=1) > 0
    
    # Optional columns: only compute is_toxic if labels exist
    if all(label in df.columns for label in label_cols):
        df['is_toxic'] = df[label_cols].sum(axis=1) > 0

    # Compute word count from lemmatized list
    df['word_count'] = df['lemmatized'].apply(len)

    return df
