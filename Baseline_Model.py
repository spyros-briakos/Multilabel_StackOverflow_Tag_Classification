# Import Useful Libraries
import pandas as pd
import numpy as np
import time
import ast
from sklearn.metrics import hamming_loss, f1_score ,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from google.colab import drive

start_time = time.time()
drive.mount('/content/drive')

# Rerieve Dataset
max = 50
path = '/content/drive/MyDrive/Satori Assignment/data/preprocessed_data_'+ str(max) +'.csv'
df = pd.read_csv(path)
df['Target'] = df['Target'].apply(ast.literal_eval)
df['Tag'] = df['Tag'].apply(ast.literal_eval)
print(df.shape)
display(df.head())

unique_tags = set([tag for sublist in df['Tag'] for tag in sublist])
print(f"Number of Unique Tags: {len(unique_tags)}")
print(unique_tags)

# Split data
def calculate_tag_combinations(temp_df):
    """
    Create a dataframe 'tag_combination', which stores the unique combination of Tags and the number of appearance
    """
    if temp_df.empty:
        print('Error: Provided DataFrame is empty!')
        return

    temp_df['Tag'] = temp_df['Tag'].apply(sorted)
    temp_df['Tag_Tuple'] = temp_df['Tag'].apply(tuple)
    tag_combinations = temp_df.groupby('Tag_Tuple').size().reset_index(name='Count')
    temp_df.drop(columns=['Tag_Tuple'], inplace=True)
    tag_combinations = tag_combinations.sort_values(by='Count', ascending=False)

    sum_count = tag_combinations['Count'].sum()
    all_tags = [tag for tag_tuple in tag_combinations['Tag_Tuple'] for tag in tag_tuple]
    num_unique_tags = len(set(all_tags))

    print(f"Number of Questions of Tag Combinations: {sum_count}")
    print(f"Number of unique tags of Tag Combinations: {num_unique_tags}")
    print(f"Minimum Frequency of Tag Combinations: {tag_combinations['Count'].iloc[-1]}\n")
    display(tag_combinations)

    return tag_combinations

original_tag_combinations = calculate_tag_combinations(df)

# Display in a proper way the distribution of Tag Combinations. This will be helpful so as to reassure in a given dataframe data are well distributed and randomly picken. This is a way to convert multilabel problem to single class problem and manually split the data on Tag Combinations and not on each single Tag.
def split_data(tag_combinations, df, create_val=True, train_ratio=0.8, val_ratio=0.1):
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    val_ratio = val_ratio if create_val else 0

    # Dicts to store counts for each combination
    train_counts = {combination: 0 for combination in tag_combinations['Tag_Tuple']}
    val_counts = {combination: 0 for combination in tag_combinations['Tag_Tuple']}
    test_counts = {combination: 0 for combination in tag_combinations['Tag_Tuple']}

    # Appropriate ratio of tag combinations for each set
    for index, row in tag_combinations.iterrows():
        combination = row['Tag_Tuple']
        count = row['Count']

        train_counts[combination] += int(count*train_ratio)
        val_counts[combination] += int(count*val_ratio)
        test_counts[combination] += count - int(count*train_ratio) - int(count*val_ratio)

    # Store data to list for each set
    train_data, val_data, test_data = [], [], []

    # Iterate over rows in df and assign to train, val, or test sets
    for index, row in df_shuffled.iterrows():
        combination = row['Tag']
        if train_counts[tuple(combination)] > 0:
            train_data.append(row)
            train_counts[tuple(combination)] -= 1
        elif test_counts[tuple(combination)] > 0:
            test_data.append(row)
            test_counts[tuple(combination)] -= 1
        elif val_counts[tuple(combination)] > 0:
            val_data.append(row)
            val_counts[tuple(combination)] -= 1

    assert len(train_data) + len(val_data) + len(test_data) == len(df), "Total number of examples in train, val, and test sets does not match the total number of examples in the original dataframe"
    return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

train_df, val_df, test_df = split_data(original_tag_combinations, df, create_val=False)
train_tag_combinations = calculate_tag_combinations(train_df)
test_tag_combinations = calculate_tag_combinations(test_df)

# TFidf Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3))
X_train = tfidf_vectorizer.fit_transform(train_df['Text'])
Y_train = np.array(train_df['Target'].tolist())
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
X_test = tfidf_vectorizer.transform(test_df['Text'])
Y_test = np.array(test_df['Target'].tolist())
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

# In order to experiment with different Sklearn models, create a generic function to monitor each algorithm's performance.
def train_and_evaluate_model(model, model_name):
    start_training = time.time()
    # Train and Predict
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Evaluation metrics
    hamming_loss_val = round(hamming_loss(Y_test, Y_pred), 2)
    micro_f1 = round(f1_score(Y_test, Y_pred, average='micro'), 2)
    macro_f1 = round(f1_score(Y_test, Y_pred, average='macro'), 2)

    print("-" * 90)
    print(f"> {model_name}")
    print(f"Hamming Loss: {hamming_loss_val}\tMicro-F1: {micro_f1}\t\tMacro-F1: {macro_f1}\t\tTime: {int((time.time()-start_training))} secs")

    return Y_pred

init = time.time()

# Logistic Regression
reg_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
train_and_evaluate_model(reg_model, 'Logistic Regression')

# Stohastic Gradient Descent
sgd_model = OneVsRestClassifier(SGDClassifier())
train_and_evaluate_model(sgd_model, 'SGD')

# Linear SVC
svc_model = OneVsRestClassifier(LinearSVC())
Y_pred = train_and_evaluate_model(svc_model, 'Linear SVC')

# Multinomial Naive Bayes
nb_model = OneVsRestClassifier(MultinomialNB(alpha=1.0))
train_and_evaluate_model(nb_model, 'Multinomial Naive Bayes')

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
train_and_evaluate_model(rf_model, 'Random Forest')

print("-" * 90)
print("\nCell Runtime:", int((time.time()-init)/60), "minutes")

# Best Classifier Cross Validation
hamming_losses, micro_f1_scores, macro_f1_scores = [], [], []
kf = KFold(n_splits=5)
# Perform manual k-fold cross-validation
for fold_idx, (train_index, test_index) in enumerate(kf.split(X_train), 1):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    Y_train_fold, Y_test_fold = Y_train[train_index], Y_train[test_index]

    # Train and Predict
    svc_model.fit(X_train_fold, Y_train_fold)
    Y_pred_fold = svc_model.predict(X_test_fold)

    # Each Fold's Metrics
    hamming_loss_fold = round(hamming_loss(Y_test_fold, Y_pred_fold), 2)
    micro_f1_fold = round(f1_score(Y_test_fold, Y_pred_fold, average='micro'), 2)
    macro_f1_fold = round(f1_score(Y_test_fold, Y_pred_fold, average='macro'), 2)

    # Store metrics
    hamming_losses.append(hamming_loss_fold)
    micro_f1_scores.append(micro_f1_fold)
    macro_f1_scores.append(macro_f1_fold)

    print("-" * 30)
    print(f"KFold {fold_idx}:")
    print(f"\tHamming Loss: {hamming_loss_fold}")
    print(f"\tMicro-F1: {micro_f1_fold}")
    print(f"\tMacro-F1: {macro_f1_fold}")

print("-" * 30 + '\n')
print("-" * 90)
print(f"CV Mean Hamming Loss: {round(np.mean(hamming_losses), 2)}\tCV Mean Micro-F1: {round(np.mean(micro_f1_scores), 2)}\t\tCV Mean Macro-F1: {round(np.mean(macro_f1_scores), 2)}")
print("-" * 90)

# Classification Report for each Tag
mlb = MultiLabelBinarizer()
tag_matrix = mlb.fit_transform(df['Tag'].tolist())
print('*'*60)
for i in range(Y_train.shape[1]):
    print(mlb.classes_[i])
    print(classification_report(Y_test[:,i], Y_pred[:,i]),'\n'+'*'*60)

print("Total time to run the notebook:", int((time.time()-start_time)/60), "minutes")