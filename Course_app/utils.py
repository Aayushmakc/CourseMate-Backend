from django.core.mail import send_mail
from django.conf import settings
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import scipy.sparse as sp
import math


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
        print("Email sent successfully!")  
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False
    
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
}

my_lematizer = WordNetLemmatizer()

def clean_and_process_data(df):
    """Clean and process DataFrame by removing duplicates and handling course IDs"""
    df = df.copy()
    def to_snake_case(name):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    df.columns = [to_snake_case(col) for col in df.columns]

    course_ids = df["course_id"]
    df_no_id = df.drop(columns=["course_id"])

    df_no_id = df_no_id.drop_duplicates()
    df_cleaned = df.loc[df_no_id.index]

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

def preprocess_courses(df):
    features_selected_for_merging = ["course_name", "course_description", "skills"]
    for col in features_selected_for_merging:
        if col not in df.columns:
            df[col] = "" 
    df["description_key_words"] = df[features_selected_for_merging].apply(lambda x: ' '.join(x.dropna()), axis=1)
    return df

def recommend_courses_by_interests(user_prefs, df):
    """
    Recommend courses based on user interests and preferences
    Args:
        user_prefs: dict with 'difficulty' and 'topics'
        df: DataFrame with course data
    """
    vectorizer = CustomTFIDFVectorizer(stop_words='english')
    course_vectors = vectorizer.fit_transform(df["description_key_words"])

    user_topics_list = user_prefs["topics"].split()
    user_vectors = [vectorizer.transform([topic]) for topic in user_topics_list]
    similarities = [cosine_similarity(user_vec, course_vectors).flatten() for user_vec in user_vectors]
    topic_to_courses = {topic: [] for topic in user_topics_list}
    
    for idx, topic in enumerate(user_topics_list):
        df["similarity"] = similarities[idx] 
        difficulty_col = 'difficulty_level' if 'difficulty_level' in df.columns else 'difficulty'
        
        top_courses = (
            df[df[difficulty_col].str.lower() == user_prefs["difficulty"].lower()]
            .sort_values(by="similarity", ascending=False)
            .head(5) 
        )
        
        columns_to_select = ["course_id"]
        if "name" in df.columns:
            columns_to_select.append("name")
        elif "course_name" in df.columns:
            columns_to_select.append("course_name")
            
        if "url" in df.columns:
            columns_to_select.append("url")
        elif "course_url" in df.columns:
            columns_to_select.append("course_url")
            
        if "rating" in df.columns:
            columns_to_select.append("rating")
        elif "course_rating" in df.columns:
            columns_to_select.append("course_rating")
            
        if "difficulty" in df.columns:
            columns_to_select.append("difficulty")
        elif "difficulty_level" in df.columns:
            columns_to_select.append("difficulty_level")

        if "university" in df.columns:
            columns_to_select.append("university")

        if "description" in df.columns:
            columns_to_select.append("description")
        elif "course_description" in df.columns:
            columns_to_select.append("course_description")

        if "skills" in df.columns:
            columns_to_select.append("skills")
            
        topic_to_courses[topic] = top_courses[columns_to_select]
    
    final_recommendations = []
    for i in range(5):  
        for topic, courses in topic_to_courses.items():
            if i < len(courses):
                final_recommendations.append(courses.iloc[i])
    result_df = pd.DataFrame(final_recommendations)
    
    column_mapping = {
        'course_id': 'course_id',
        'name': 'course_name',
        'course_name': 'course_name',
        'url': 'course_url',
        'course_url': 'course_url',
        'rating': 'course_rating',
        'course_rating': 'course_rating',
        'difficulty': 'difficulty_level',
        'difficulty_level': 'difficulty_level',
        'university': 'university',
        'description': 'course_description',
        'course_description': 'course_description',
        'skills': 'skills'
    }
  
    rename_dict = {col: column_mapping[col] for col in result_df.columns if col in column_mapping}
    result_df = result_df.rename(columns=rename_dict)
    
    return result_df

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
                    filtered_df['difficulty_level'].str.lower() == difficulty_level.lower()
                ]
        
        if min_rating is not None:
            min_rating = float(min_rating)
            filtered_df = filtered_df[filtered_df['course_rating'] >= min_rating]
            
        if max_rating is not None:
            max_rating = float(max_rating)
            filtered_df = filtered_df[filtered_df['course_rating'] <= max_rating]
            
        if filtered_df.empty:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in filtering: {str(e)}")
        return pd.DataFrame()
            
    return filtered_df



def books_id_recommended(description, vectorizer, vectors, number_of_recommendation=5, return_scores=False):
    try:
        print("DEBUG: Starting books_id_recommended")  
        min_similarity = 0.075  
        
        description = [PreprocessTexte(description)]
        vect = vectorizer.transform(description)
        
        similars_vectors = cosine_similarity(vect, vectors)[0]
        print(f"DEBUG: Similarity scores range: {similars_vectors.min():.3f} to {similars_vectors.max():.3f}")  
        
        similars_vectors = np.array(similars_vectors)
        
        ordered_indices = np.argsort(similars_vectors)
        sorted_scores = similars_vectors[ordered_indices][::-1]
        valid_recommendations = np.sum(sorted_scores >= min_similarity)
        print(f"DEBUG: Found {valid_recommendations} recommendations above threshold {min_similarity}")  
        
        if valid_recommendations == 0:
            print("DEBUG: No recommendations meet similarity threshold, using top 5")  
            n_recommendations = min(5, len(sorted_scores))
        else:
            n_recommendations = min(number_of_recommendation, max(1, valid_recommendations))
        
        best_indices = ordered_indices[::-1][:n_recommendations]
        print(f"DEBUG: Returning {len(best_indices)} indices") 
        
        if return_scores:
            return best_indices.tolist(), sorted_scores[:n_recommendations].tolist()
        return best_indices.tolist()
        
    except Exception as e:
        print(f"Error in books_id_recommended: {str(e)}")
        import traceback
        traceback.print_exc()  
        return ([], []) if return_scores else []

def find_top_k_indices(df, k):
    """Find top k indices from similarity matrix"""
    top_k_indices = np.unravel_index(np.argsort(df.values, axis=None)[-k:], df.shape)
    return list(zip(*top_k_indices))

def process_user_profile(user_id, vectorizer):
    from .models import UserProfile
    
    user_profiles = UserProfile.objects.filter(user_id=user_id)
    user_df = pd.DataFrame.from_records(user_profiles.values())
    
    if user_df.empty:
        raise ValueError(f"No profile found for user {user_id}")

    user_df.drop_duplicates(inplace=True)
    features_selected_for_merging = ["course_name", "course_description", "skills"]
    user_df["description_key_words"] = user_df[features_selected_for_merging].apply(
        lambda x: ' '.join(str(val) for val in x), axis=1
    )
    
    user_df["description_key_words"] = user_df["description_key_words"].apply(PreprocessTexte)
    user_vectors = vectorizer.transform(user_df["description_key_words"]).toarray()

    user_rating = user_df['course_rating']
    user_rating_scaled = (user_rating - 1) / 4
    user_rating_scaled = user_rating_scaled.tolist()


    user_vectors_with_review = user_vectors * np.array(user_rating_scaled).reshape(-1, 1)

    return user_df, user_vectors, user_vectors_with_review

def recommend_courses(user_id, main_vectors, df):
    vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
    main_vectors = vectorizer.fit_transform(df["description_key_words"])
    
    user_df, user_vectors, user_vectors_with_review = process_user_profile(user_id, vectorizer)

    
    similars_vectors = cosine_similarity(main_vectors, user_vectors_with_review)
    similars_vectors_df = pd.DataFrame(similars_vectors)

   
    df_course_id_to_index = {course_id: idx for idx, course_id in enumerate(df['course_id'])}
    user_course_id_to_index = {course_id: idx for idx, course_id in enumerate(user_df['course_id'])}
    
    matching_indices = [
        (df_course_id_to_index[course_id], user_course_id_to_index[course_id])
        for course_id in df['course_id']
        if course_id in user_course_id_to_index
    ]

    top_indices = find_top_k_indices(similars_vectors_df, 21)
    filtered_top_indices = [index for index in top_indices if index not in matching_indices]
    top_10_indices = filtered_top_indices[:10]
    top_10_rows = [row for row, col in top_10_indices[:10]]

    return top_10_rows




























def recommend_courses(user_interests, df):
    """
    Recommend courses based on user interests using TF-IDF and cosine similarity
    Args:
        user_interests: string containing user interests and preferences
        df: DataFrame with course data
    Returns:
        DataFrame with recommended courses
    """
    try:
        print("DEBUG: Starting recommend_courses")  
        print(f"DEBUG: Input DataFrame shape: {df.shape}")  
        
       
        user_interests = PreprocessTexte(user_interests)
        print(f"DEBUG: Preprocessed user interests: {user_interests[:100]}...")  
        
        
        vectorizer = CustomTFIDFVectorizer(stop_words='english')
        
        
        if 'description_key_words' not in df.columns:
            print("DEBUG: Creating description_key_words column")  
            df = preprocess_courses(df)
        
        print(f"DEBUG: Number of non-empty description_key_words: {df['description_key_words'].str.strip().str.len().gt(0).sum()}")  
        
        course_vectors = vectorizer.fit_transform(df["description_key_words"])
        print(f"DEBUG: Created course vectors with shape: {course_vectors.shape}")  
        recommended_indices, similarity_scores = books_id_recommended(
            user_interests, 
            vectorizer, 
            course_vectors,
            number_of_recommendation=20,
            return_scores=True
        )
        
        print(f"DEBUG: Got {len(recommended_indices)} recommended indices with scores: {similarity_scores[:5]}")  
        
        if not recommended_indices:
            print("DEBUG: No recommended indices returned")  
            return pd.DataFrame()
            
        
        recommendations = df.iloc[recommended_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        
        
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        print(f"DEBUG: Final recommendations shape: {recommendations.shape}")  
        
       
        recommendations = recommendations.drop(['similarity_score', 'description_key_words'], axis=1, errors='ignore')
        
        return recommendations
        
    except Exception as e:
        print(f"Error in recommend_courses: {str(e)}")
        import traceback
        traceback.print_exc()  
        return pd.DataFrame()