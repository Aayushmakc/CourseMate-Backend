from django.core.mail import send_mail
from django.conf import settings
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import scipy.sparse as sp
import math


# This is for emailsetup

def send_welcome_email(user_email, first_name):
    subject = 'Welcome to CourseMate!'
    message = f"""
    Hi {first_name},
    
    Welcome to our Course Recommendation System!🎉 
    We're excited to have you join us.
    
    Start exploring courses that match your interests.
    
    Best regards,
    CourseMate Team
    """
    print(f"Attempting to send email to {user_email}") 
    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[user_email],
            fail_silently=False,
        ) 
        print("Email sent successfully!")  # Debug print
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False
    
# utils.py
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
import math
import re
from nltk.stem import WordNetLemmatizer

# Predefined stopwords list
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    # ... (same as your original stopwords)
}

my_lematizer = WordNetLemmatizer()

def clean_and_process_data(df):
    # Rename columns
    def rename_col(col_name):
        return '_'.join(col_name.split(' '))

    df.columns = [rename_col(col) for col in df.columns]

    # Separate course_id before dropping duplicates
    course_ids = df["course_id"] if "course_id" in df.columns else None

    # Drop duplicates ignoring course_id
    df_no_id = df.drop(columns=["course_id"]) if "course_id" in df.columns else df.copy()
    df_no_id = df_no_id.drop_duplicates()

    # Reattach course_id (keeping only the first occurrence of each unique row)
    if course_ids is not None:
        df_cleaned = pd.concat([course_ids[df_no_id.index], df_no_id], axis=1)
    else:
        df_cleaned = df_no_id

    return df_cleaned

def PreprocessTexte(text):
    cleaned_text = re.sub(r'-',' ',str(text)) 
    cleaned_text = re.sub(r'https?://\S+|www\.\S+|http?://\S+',' ',cleaned_text) 
    cleaned_text = re.sub(r'<.*?>',' ',cleaned_text) 
    cleaned_text = re.sub(r'[0-9]', '', cleaned_text)
    cleaned_text = re.sub(r"\([^()]*\)", "", cleaned_text)
    cleaned_text = re.sub('@\S+', '', cleaned_text)  
    cleaned_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleaned_text)  
    cleaned_text = re.sub(r'ML',' Machine Learning ',cleaned_text) 
    cleaned_text = re.sub(r'DL',' Deep Learning ',cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.split()
    cleaned_text = ' '.join([my_lematizer.lemmatize(word) for word in cleaned_text])
    return cleaned_text

class CustomTFIDFVectorizer:
    """Custom TF-IDF Vectorizer implementation"""
    def __init__(self, max_features=None, stop_words="english"):
        self.max_features = max_features
        self.vocab = None
        self.idf_values = None
        self.stop_words = ENGLISH_STOPWORDS if stop_words == 'english' else set(stop_words)

    def fit_transform(self, corpus):
        """Computes TF-IDF and returns a CSR sparse matrix"""
        tokenized_corpus = []
        for doc in corpus:
            tokens = [word.lower() for word in str(doc).split() 
                     if word.lower() not in self.stop_words and word.isalnum()]
            tokenized_corpus.append(tokens)
        
        word_counts = Counter()
        for tokens in tokenized_corpus:
            word_counts.update(tokens)

        vocabulary = ([word for word, _ in word_counts.most_common(self.max_features)] 
                     if self.max_features else list(word_counts.keys()))

        self.vocab = {word: i for i, word in enumerate(vocabulary)}
        
        num_docs = len(corpus)
        doc_freq = Counter(word for tokens in tokenized_corpus 
                         for word in set(tokens) if word in self.vocab)
        self.idf_values = {word: math.log((1 + num_docs) / (1 + df)) + 1 
                          for word, df in doc_freq.items()}

        rows, cols, values = [], [], []
        for i, tokens in enumerate(tokenized_corpus):
            term_frequencies = Counter(tokens)
            for word, count in term_frequencies.items():
                if word in self.vocab:
                    tf = count / len(tokens)
                    idf = self.idf_values[word]
                    rows.append(i)
                    cols.append(self.vocab[word])
                    values.append(tf * idf)

        return sp.csr_matrix((values, (rows, cols)), 
                           shape=(num_docs, len(self.vocab)))

    def transform(self, corpus):
        """Transform new documents using learned vocabulary"""
        tokenized_corpus = []
        for doc in corpus:
            tokens = [word.lower() for word in str(doc).split() 
                     if word.lower() not in self.stop_words and word.isalnum()]
            tokenized_corpus.append(tokens)

        rows, cols, values = [], [], []
        for i, tokens in enumerate(tokenized_corpus):
            term_frequencies = Counter(tokens)
            for word, count in term_frequencies.items():
                if word in self.vocab:
                    tf = count / len(tokens)
                    idf = self.idf_values.get(word, 0)
                    rows.append(i)
                    cols.append(self.vocab[word])
                    values.append(tf * idf)

        return sp.csr_matrix((values, (rows, cols)), 
                           shape=(len(corpus), len(self.vocab)))

def cosine_similarity(matrix1, matrix2=None):
    """Compute cosine similarity between matrices"""
    if matrix2 is None:
        matrix2 = matrix1

    if not sp.issparse(matrix1):
        matrix1 = sp.csr_matrix(matrix1)
    if not sp.issparse(matrix2):
        matrix2 = sp.csr_matrix(matrix2)

    dot_product = matrix1 @ matrix2.T
    norm1 = np.sqrt(matrix1.multiply(matrix1).sum(axis=1))
    norm2 = np.sqrt(matrix2.multiply(matrix2).sum(axis=1)).T
    
    norm1[norm1 == 0] = 1e-10
    norm2[norm2 == 0] = 1e-10
    
    return (dot_product / (norm1 @ norm2)).toarray()

def filter_dataframe_function(df, difficulty_level=None, min_rating=None, max_rating=None):
    """Filter courses based on difficulty and rating range"""
    filtered_df = df.copy()
    
    try:
        if difficulty_level:
            valid_difficulties = ["beginner", "intermediate", "advanced"]
            if difficulty_level.lower() in valid_difficulties:
                filtered_df = filtered_df[
                    filtered_df['difficulty'].str.lower() == difficulty_level.lower()
                ]
        
        if min_rating is not None:
            min_rating = float(min_rating)
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
            
        if max_rating is not None:
            max_rating = float(max_rating)
            filtered_df = filtered_df[filtered_df['rating'] <= max_rating]
            
        if filtered_df.empty:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in filtering: {str(e)}")
        return pd.DataFrame()
            
    return filtered_df

def books_id_recommended(description, vectorizer, vectors, number_of_recommendation=5):
    min_similarity = 0.075
    description = [PreprocessTexte(description)]
    vect = vectorizer.transform(description)
    similars_vectors = cosine_similarity(vect, vectors)[0]
    ordered_similars_vectors = list(similars_vectors.argsort())
    x = list(similars_vectors)
    x.sort(reverse=True)
    a = 1
    for i in x[1:number_of_recommendation+5]:
        if i >= min_similarity:
            a = a+1
    reverse_ordered_similars_vectors = [index for index in reversed(ordered_similars_vectors)]
    best_indexs = reverse_ordered_similars_vectors[1:min(number_of_recommendation,a)]
    return best_indexs

def find_top_k_indices(df, k):
    """Find top k indices from similarity matrix"""
    top_k_indices = np.unravel_index(np.argsort(df.values, axis=None)[-k:], df.shape)
    return list(zip(*top_k_indices))

def process_user_profile(user_id, vectorizer):
    from .models import UserProfile
    
    # Get user profile from database
    user_profiles = UserProfile.objects.filter(user_id=user_id)
    
    # Convert to DataFrame
    user_df = pd.DataFrame.from_records(user_profiles.values())
    
    if user_df.empty:
        raise ValueError(f"No profile found for user {user_id}")

    # Drop duplicates
    user_df.drop_duplicates(inplace=True)

    # Merge key features into description_key_words
    features_selected_for_merging = ["course_name", "course_description", "skills"]
    user_df["description_key_words"] = user_df[features_selected_for_merging].apply(
        lambda x: ' '.join(str(val) for val in x), axis=1
    )
    
    # Preprocess text
    user_df["description_key_words"] = user_df["description_key_words"].apply(PreprocessTexte)

    # Use the same vectorizer that was used for the main corpus
    user_vectors = vectorizer.transform(user_df["description_key_words"]).toarray()

    # Scale ratings
    user_rating = user_df['course_rating']
    user_rating_scaled = (user_rating - 1) / 4
    user_rating_scaled = user_rating_scaled.tolist()

    # Weight user vectors by scaled ratings
    user_vectors_with_review = user_vectors * np.array(user_rating_scaled).reshape(-1, 1)

    return user_df, user_vectors, user_vectors_with_review

def recommend_courses(user_id, main_vectors, df):
    # Create and fit vectorizer on main corpus first
    vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
    main_vectors = vectorizer.fit_transform(df["description_key_words"])
    
    # Process the user profile using the same vectorizer
    user_df, user_vectors, user_vectors_with_review = process_user_profile(user_id, vectorizer)

    # Calculate cosine similarity
    similars_vectors = cosine_similarity(main_vectors, user_vectors_with_review)
    similars_vectors_df = pd.DataFrame(similars_vectors)

    # Find indices of matching courses
    df_course_id_to_index = {course_id: idx for idx, course_id in enumerate(df['Course_ID'])}
    user_course_id_to_index = {course_id: idx for idx, course_id in enumerate(user_df['course_id'])}
    
    matching_indices = [
        (df_course_id_to_index[course_id], user_course_id_to_index[course_id])
        for course_id in df['Course_ID']
        if course_id in user_course_id_to_index
    ]

    # Find top indices
    top_indices = find_top_k_indices(similars_vectors_df, 21)
    filtered_top_indices = [index for index in top_indices if index not in matching_indices]
    top_10_indices = filtered_top_indices[:10]
    top_10_rows = [row for row, col in top_10_indices[:10]]

    return top_10_rows