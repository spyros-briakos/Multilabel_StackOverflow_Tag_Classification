# Import Useful Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from wordcloud import WordCloud
from google.colab import drive
import contractions

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Retrieve and display data
questions_path = '/content/drive/MyDrive/Satori Assignment/data/Questions.csv'
tags_path = '/content/drive/MyDrive/Satori Assignment/data/Tags.csv'

try:
    questions_df = pd.read_csv(questions_path, encoding='utf-8')
except UnicodeDecodeError:
    questions_df = pd.read_csv(questions_path, encoding='latin1')

try:
    tags_df = pd.read_csv(tags_path, encoding='utf-8')
except UnicodeDecodeError:
    tags_df = pd.read_csv(tags_path, encoding='latin1')

# Fill tha NaN values with a space, in order to avoid future errors on merging."""
questions_df.fillna(' ', inplace=True)

# Merge dataframes (plus aggregation)
tags_df['Tag'] = tags_df['Tag'].astype(str)
merged_df = pd.merge(questions_df, tags_df, how='inner', on='Id')
merged_df = merged_df.groupby(['Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title', 'Body']).agg({'Tag': lambda x: ', '.join(x)}).reset_index()
merged_df['Tag'] = merged_df['Tag'].apply(lambda x: x.split(', '))

duplicates = merged_df[merged_df.duplicated(subset=['Id'], keep=False)]
if duplicates.empty == False: print('Error: Found duplicate row!')

# Preprocessing
# Positive Score Questions
merged_df = merged_df[merged_df['Score'] > 0]

# Most Frequent Tag Combinations
merged_df['Tag'] = merged_df['Tag'].apply(sorted)
merged_df['Tag_Tuple'] = merged_df['Tag'].apply(tuple)
tag_combinations = merged_df.groupby('Tag_Tuple').size().reset_index(name='Count')
tag_combinations = tag_combinations.sort_values(by='Count', ascending=False)

max = 50
top_tag_combinations = tag_combinations.head(max)
sum_count = top_tag_combinations['Count'].sum()
all_tags = [tag for tag_tuple in top_tag_combinations['Tag_Tuple'] for tag in tag_tuple]
num_unique_tags = len(set(all_tags))
top_tag_tuples = set(top_tag_combinations['Tag_Tuple'])
merged_df = merged_df[merged_df['Tag_Tuple'].apply(lambda x: x in top_tag_tuples)]

# Give emphasis on Title
P = 3
merged_df['Text'] = (merged_df['Title'] + " ") * P + merged_df['Body']
final_df = merged_df[['Text', 'Tag']].copy()

def calculate_unique_tags(temp_df):
    all_tags = [tag for tags_list in temp_df['Tag'] for tag in tags_list]
    total_unique_tags = len(set(all_tags))
    print("Total number of unique tags:", total_unique_tags)
calculate_unique_tags(final_df)

# Preprocessing pipeline
def replace_word_starting_with(text, prefixes):
    words = text.split()
    for i in range(len(words)):
        if words[i].startswith('javascript'):
            words[i] = 'javascript'
        if words[i].startswith('c++'):
            words[i] = 'c++'
        if words[i].startswith('c#'):
            words[i] = 'c#'
        if words[i].startswith('c') or words[i].startswith('r') or words[i].startswith('java'):
            continue
        for prefix in prefixes:
            if words[i].startswith(prefix):
                words[i] = prefix
                break
    return ' '.join(words)

def clean_text(text, for_tags=False):
    text = text.lower()

    if for_tags == False and debugging: print("BEFORE:",text)
    if for_tags == False:
        text = replace_word_starting_with(text,exclusion_list)
    if for_tags == False and debugging: print("AFTER:",text)

    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub(r"\b\w+\.jpg\b", " ", text)
    text = re.sub(r"\b0x[0-9a-f]+\b","", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"(c)/+(c\+\+)", r"\1 \2", text)
    text = re.sub(r"\bjs\b", 'javascript', text)
    text = re.sub("\s+", " ", text) # Extra spaces
    text = re.sub(r"<.*?>", '', text) # HTML tags
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", '', text) # IPV4 addresses
    if for_tags == False:
        text = re.sub(r'http\S+', '', text) # URLs
        text = contractions.fix(text)
    text = text.strip(' ')
    return text

def tokenize_except(text):
    tokens = text.split()
    tokenized_text = []
    for token in tokens:
        if token not in exclusion_list:
            tokenized_text.extend(word_tokenize(token))
        else:
            tokenized_text.append(token)
    return tokenized_text

def only_puncts(token):
    for char in token:
        if char not in string.punctuation:
            return False
    return True

all_tags = [tag for tags_list in final_df['Tag'] for tag in tags_list]
exclusion_list = set([clean_text(tag,True) for tag in all_tags])
debugging = False

def preprocess_pipeline(text):
    """
    Lowercase -> Cleaning-> Tokenization -> Punctuation Removal ->
    Stop Words Removal -> Lemmatization -> Joining
    """

    text = clean_text(text)

    if debugging:  print(text)

    tokens = tokenize_except(text)

    if debugging:  print(tokens)

    keep_punctuation = '.-#+'
    punctuation_to_remove = set(string.punctuation) - set(keep_punctuation)
    tokens = [token for token in tokens if token not in punctuation_to_remove]

    if debugging:  print(tokens)

    special_cases = ["'s", "'t", "e.g"]
    stop_words = set(stopwords.words('english') + special_cases)
    tokens = [token for token in tokens if (token not in stop_words) and (token.isdigit() == False)]

    if debugging:  print(tokens)

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in exclusion_list]

    if debugging:  print(tokens)

    tokens = [token for token in tokens if only_puncts(token) == False]

    if debugging:  print(tokens)

    text = ' '.join(tokens)

    return text

final_df['Text'] = final_df['Text'].apply(preprocess_pipeline)

mlb = MultiLabelBinarizer(sparse_output=False)
tag_matrix = mlb.fit_transform(final_df['Tag'].tolist())
final_df['Target'] = tag_matrix.tolist()

# Save dataframe to CSV
final_df.to_csv('/content/drive/MyDrive/Satori Assignment/data/preprocessed_data.csv', index=False)