# Import Useful Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from wordcloud import WordCloud
from google.colab import drive

!pip install contractions
import contractions
pd.options.mode.chained_assignment = None

start_time = time.time()
drive.mount('/content/drive')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## Retrieve and display data
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

display(questions_df.info())

# Fill tha NaN values with a space, in order to avoid future errors on merging."""
questions_df.fillna(' ', inplace=True)
display(questions_df.head())
display(tags_df.info())
display(tags_df.head())

# Merge dataframes (plus aggregation)
tags_df['Tag'] = tags_df['Tag'].astype(str)
merged_df = pd.merge(questions_df, tags_df, how='inner', on='Id')
merged_df = merged_df.groupby(['Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title', 'Body']).agg({'Tag': lambda x: ', '.join(x)}).reset_index()
merged_df['Tag'] = merged_df['Tag'].apply(lambda x: x.split(', '))
merged_df.head()

print('Number of training examples:', merged_df.shape[0])

duplicates = merged_df[merged_df.duplicated(subset=['Id'], keep=False)]
if duplicates.empty == False: print('Error: Found duplicate row!')

# Score
merged_df['Score'].describe()
print('Only ' + str((merged_df['Score']<0).sum()) + ' questions have negative Score.')
print('Only ' + str((merged_df['Score']==0).sum()) + ' questions have zero Score.')
print('Only ' + str((merged_df['Score']>0).sum()) + ' questions have positive Score.')
print('Only ' + str((merged_df['Score']>5).sum()) + ' questions have Score greater than 5.')

merged_df['Tag_Length'] = merged_df['Tag'].apply(len)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Tag_Length', y='Score', data=merged_df)
plt.yscale('log')
plt.title('Relationship between Tag Length and Score')
plt.xlabel('Tag Length')
plt.ylabel('Score')
plt.grid(True)
plt.show()

mean_scores = merged_df.groupby('Tag_Length')['Score'].mean()
plt.figure(figsize=(10, 6))
mean_scores.plot(kind='bar')
plt.title('Mean Score for Each Tag Length')
plt.xlabel('Tag Length')
plt.ylabel('Mean Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Creation and Closed Dates
print('Only ' + str((merged_df['ClosedDate']!=' ').sum()) + ' questions contain Closed Date.')

merged_df['CreationDate'] = pd.to_datetime(merged_df['CreationDate'])
monthly_counts = merged_df.resample('M', on='CreationDate').size()
monthly_counts.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Monthly Distribution of Questions')
plt.xlabel('Month')
plt.ylabel('Number of Posts')
plt.grid(True)
plt.show()

hourly_counts = merged_df.groupby(merged_df['CreationDate'].dt.hour).size()
hourly_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Hourly Distribution of Questions')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

daily_counts = merged_df.groupby(merged_df['CreationDate'].dt.dayofweek).size()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_counts.index = days
daily_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Daily Distribution of Questions')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Questions')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Explore Tags
N = 100
tag_frequencies = tags_df['Tag'].value_counts().head(N)
plt.figure(figsize=(15, 10))
plt.bar(tag_frequencies.index, tag_frequencies, color='skyblue')
plt.xlabel('Tag')
plt.ylabel('Tag Frequency')
plt.title('Tag Frequency of the {} Most Frequent Tags'.format(N))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# WordCloud
tag_frequencies = tags_df['Tag'].value_counts()
wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(tag_frequencies.head(1000))
plt.figure(figsize=(15, 10))
plt.title('WordCloud of Top 1000 Frequent Tags')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Questions Number of Tags Frequency
statistics = merged_df['Tag_Length'].value_counts().sort_index()
plt.bar(statistics.index, statistics.values, color='skyblue')
plt.xlabel('Number of Tags on each Question')
plt.ylabel('Frequency')
plt.title('Questions Number of Tags Frequency')
plt.xticks(statistics.index)
plt.show()
print('The Average Number of Tags per Question is',"{:.2f}".format(sum(merged_df['Tag_Length'].values)*1.0/(len(merged_df['Tag_Length'].values)),2), 'tags')

# Coverage of Top N Tags against all Tags
total_unique_tags = len(tags_df['Tag'].unique())
print("Total Unique Tags:", total_unique_tags)

tag_frequencies = tags_df['Tag'].value_counts()
N_values = [20,500,1000,2000,5000]
coverage_values = []
total_entries = len(tags_df)
for N in N_values:
    top_N_tags = tag_frequencies.head(N)
    total_frequency = top_N_tags.sum()
    coverage = (total_frequency / total_entries) * 100
    coverage_values.append(coverage)

plt.figure(figsize=(12, 10))
plt.plot(N_values, coverage_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Top Tags Retained')
plt.ylabel('Percentage Coverage')
plt.title('Percentage Coverage of All Tags vs. Number of Top Tags Retained')
plt.grid(True)
plt.xticks(N_values)
plt.show()

# Coverage of Questions if we retain Top N Tags
questions_coverage = []
total_questions = len(merged_df)

for N in N_values:
    top_N_tags = tag_frequencies.head(N).index.tolist()
    # Filter to include only questions with top N tags
    filtered_df = merged_df[merged_df['Tag'].apply(lambda x: any(tag in x for tag in top_N_tags))]
    unique_questions = filtered_df['Id'].nunique()
    coverage_percentage = (unique_questions / total_questions) * 100
    questions_coverage.append(coverage_percentage)

plt.figure(figsize=(8, 6))
plt.plot(N_values, questions_coverage, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Top Tags Retained')
plt.ylabel('Percentage of Questions Covered')
plt.title('Percentage of Questions Covered vs. Number of Top Tags Retained')
plt.grid(True)
plt.xticks(N_values)
plt.show()

# Preprocessing
# Positive Score Questions
merged_df = merged_df[merged_df['Score'] > 0]
print('Number of Questions with positive Score:', merged_df.shape[0])

# Most Frequent Tag Combinations
merged_df['Tag'] = merged_df['Tag'].apply(sorted)
merged_df['Tag_Tuple'] = merged_df['Tag'].apply(tuple)
tag_combinations = merged_df.groupby('Tag_Tuple').size().reset_index(name='Count')
tag_combinations = tag_combinations.sort_values(by='Count', ascending=False)
tag_combinations

F = 20
top_F_df = tag_combinations.head(F).copy()
top_F_df['Tag_Tuple_str'] = top_F_df['Tag_Tuple'].astype(str)

plt.figure(figsize=(16, 8))
plt.bar(top_F_df['Tag_Tuple_str'], top_F_df['Count'])
plt.title(f'Top {F} Frequent Tag Combinations')
plt.xlabel('Tag Tuple')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

max = 50
top_tag_combinations = tag_combinations.head(max)
sum_count = top_tag_combinations['Count'].sum()
all_tags = [tag for tag_tuple in top_tag_combinations['Tag_Tuple'] for tag in tag_tuple]
num_unique_tags = len(set(all_tags))
print(f"Number of Questions of Top {max} Tag Combinations: {sum_count}")
print(f"Number of unique tags of Top {max} Tag Combinations: {num_unique_tags}")
print(f"Minimum Frequency of Top {max} Tag Combinations: {top_tag_combinations['Count'].iloc[-1]}\n")

questions_coverage = []
total_questions = len(merged_df)
D_values = range(100,600,100)
for D in D_values:
    top_D_tags_count = tag_combinations.head(D)['Count'].sum()
    coverage_percentage = (top_D_tags_count / total_questions) * 100
    questions_coverage.append(coverage_percentage)

plt.figure(figsize=(8, 6))
plt.plot(D_values, questions_coverage, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Top Tag Combinations')
plt.ylabel('Percentage of Questions Covered')
plt.title('Percentage of Questions Covered vs. Number of Top Tag Combinations')
plt.grid(True)
plt.xticks(D_values)
plt.show()

top_tag_tuples = set(top_tag_combinations['Tag_Tuple'])
merged_df = merged_df[merged_df['Tag_Tuple'].apply(lambda x: x in top_tag_tuples)]

# Give emphasis on Title
P = 3
merged_df['Text'] = (merged_df['Title'] + " ") * P + merged_df['Body']
final_df = merged_df[['Text', 'Tag']].copy()
display(final_df.head(6))

def calculate_unique_tags(temp_df):
    all_tags = [tag for tags_list in temp_df['Tag'] for tag in tags_list]
    total_unique_tags = len(set(all_tags))
    print("Total number of unique tags:", total_unique_tags)

calculate_unique_tags(final_df)
print('Number of Questions:', final_df.shape[0])

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
    text = re.sub(r'\b0x[0-9a-f]+\b','', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'(c)/+(c\+\+)', r'\1 \2', text)
    text = re.sub(r'\bjs\b', 'javascript', text)
    text = re.sub('\s+', " ", text) # Extra spaces
    text = re.sub(r'<.*?>', '', text) # HTML tags
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '', text) # IPV4 addresses
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

K=101
random_three_texts = final_df['Text'].iloc[K:K+1].apply(preprocess_pipeline)
print(final_df['Tag'].iloc[K:K+1])
print(final_df['Text'].iloc[K:K+1])
print(random_three_texts)

k = time.time()
final_df['Text'] = final_df['Text'].apply(preprocess_pipeline)
print("Total time :", int((time.time()-k)/60), "minutes\n")

example = """ http https://dsads c/c++ c#? c++? visual-studio-2008 visual-c++ c++ c# r qt c#? c++? my spyro's ball and colleaques aren't doing well ,,,,
              we don't         js javascript PERFECT them. 321 3213 1 maggania#@#@!#!@ wewe##
              dsad.jpg <a href="http://svnbook.red-bean.com/en/1.8/svn.branchmerge.html" rel="nofollow"></a>
          """
print(preprocess_pipeline(example))

# Explore last-stage data
final_df['Num_Tags'] = final_df['Tag'].apply(lambda x: len(x))
final_df['Num_Words'] = final_df['Text'].apply(lambda x: len(x.split()))
tag_counts = (final_df['Num_Tags'].value_counts(normalize=True) * 100).sort_index().tolist()
labels = ['1 Tag', '2 Tags', '3 Tags', '4 Tags']
explode = (0.05, 0, 0, 0)
plt.figure(figsize=(10, 6))
plt.pie(tag_counts, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Questions Percentage by Number of Tags')
plt.axis('equal')
plt.show()

print(final_df['Num_Tags'].describe(),'\n')

plt.figure(figsize=(8, 6))
sns.histplot(final_df['Num_Tags'], bins=10, kde=True, color='salmon')
plt.title('Distribution of Number of Tags')
plt.xlabel('Number of Tags')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(final_df['Num_Tags'], final_df['Num_Words'])
plt.title('Correlation between Number of Tags and Number of Words')
plt.xlabel('Number of Tags')
plt.ylabel('Number of Words')
plt.show()

print("Statistics about the number of words in the 'Text' column:")
print(final_df['Num_Words'].describe(),'\n')

plt.figure(figsize=(8, 6))
sns.histplot(final_df['Num_Words'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Number of Words')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

mlb = MultiLabelBinarizer(sparse_output=False)
tag_matrix = mlb.fit_transform(final_df['Tag'].tolist())
final_df['Target'] = tag_matrix.tolist()
print("Shape of tag matrix:", tag_matrix.shape)

w = 10
print(final_df['Tag'].iloc[w])
tag_matrix[w]

# Save dataframe to CSV
final_df.to_csv('/content/drive/MyDrive/Satori Assignment/data/preprocessed_data.csv', index=False)
print("Total time to run the notebook:", int((time.time()-start_time)/60), "minutes")