# Import Required Libraries

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')


# Step 1: Load and Preprocess Knowledge Graph
knowledge_graph_path = "./data/Knowledge_graph.csv"
df = pd.read_csv(knowledge_graph_path)
df.drop_duplicates(inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

# Replace unwanted prefixes
prefix_replacements = {
    'Compound::': 'Compound_',
    'protein::': 'protein_',
    'Disease::': 'Disease_'
}
for key, value in prefix_replacements.items():
    df['source'] = df['source'].str.replace(key, value)
    df['target'] = df['target'].str.replace(key, value)

print("Knowledge Graph shape:", df.shape)

# Step 2: Load Positive and Negative Train-Test Data
positive_train_path = "./data/positive_test_df.csv"
negative_train_path = "./data/negative_test_df.csv"
positive_df = pd.read_csv(positive_train_path)
negative_df = pd.read_csv(negative_train_path)

# Rename columns for consistency
positive_df.rename(columns={"drug_id": "source", "ind_id": "target"}, inplace=True)
negative_df.rename(columns={"drug_id": "source", "ind_id": "target"}, inplace=True)

positive_df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
negative_df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

# Display dataset shapes
print("Positive Train-Test shape:", positive_df.shape)
print("Negative Train-Test shape:", negative_df.shape)

# Step 3: Filter Rows to Delete from Knowledge Graph
# Create composite keys for matching
df['composite_key'] = df['source'] + '_' + df['target']
positive_df['composite_key'] = positive_df['source'] + '_' + positive_df['target']
negative_df['composite_key'] = negative_df['source'] + '_' + negative_df['target']

# Identify rows to delete
df_to_delete_positive = df[(df['relation'] == 'has_approved_interaction') &
                           (df['composite_key'].isin(positive_df['composite_key']))]

df_to_delete_negative = df[(df['relation'] == 'has_side_effect') &
                           (df['composite_key'].isin(negative_df['composite_key']))]

df_to_delete = pd.concat([df_to_delete_positive, df_to_delete_negative])
df_cleaned = df.drop(df_to_delete.index)
df_cleaned.drop(columns=['composite_key'], inplace=True)

# Save cleaned DataFrame
df_cleaned_path = "./output/train_df_Word2Vec.csv"
df_cleaned.to_csv(df_cleaned_path, index=False)
print("Cleaned Knowledge Graph shape:", df_cleaned.shape)

# Step 4: Prepare Sentences for Word2Vec
sentences = [f"{row['source']} {row['relation']} {row['target']} ." for _, row in df_cleaned.iterrows()]
sentences_tokenized = [word_tokenize(sent) for sent in sentences]

# Step 5: Train Word2Vec Model
model = Word2Vec(vector_size=650, window=2, sg=0, min_count=1, epochs=100, alpha=0.001)
model.build_vocab(sentences_tokenized, progress_per=1500)
model.train(sentences_tokenized, total_examples=model.corpus_count, epochs=500, report_delay=1)
model.save("./output/model1.model")

# Step 6: Prepare X.npy and y.npy for Training
positive_train_path = "./data/positive_train_df.csv"
negative_train_path = "./data/negative_train_df.csv"
has = pd.read_csv(positive_train_path)
hasnt = pd.read_csv(negative_train_path)

# Rename columns and clean prefixes
has.rename(columns={"source": "drug_id", "target": "ind_id"}, inplace=True)
hasnt.rename(columns={"source": "drug_id", "target": "ind_id"}, inplace=True)
for key, value in prefix_replacements.items():
    has['drug_id'] = has['drug_id'].str.replace(key, value)
    has['ind_id'] = has['ind_id'].str.replace(key, value)
    hasnt['drug_id'] = hasnt['drug_id'].str.replace(key, value)
    hasnt['ind_id'] = hasnt['ind_id'].str.replace(key, value)

# Concatenate positive and negative data
frames = [has, hasnt]
f = pd.concat(frames)

# Generate feature vectors
X = [
    np.concatenate((model.wv[row['drug_id']].reshape((1, 650)),
                    model.wv[row['ind_id']].reshape((1, 650))), axis=None)
    for _, row in f.iterrows()
]

# Create labels
y = np.zeros(len(X))
y[:len(has)] = 1

np.save("./output/X.npy", X)
np.save("./output/y.npy", y)

# Step 7: Prepare Test Data
positive_test_path = "./data/positive_test_df.csv"
negative_test_path = "./data/negative_test_df.csv"
positive_test_df = pd.read_csv(positive_test_path)
negative_test_df = pd.read_csv(negative_test_path)

# Add positive and negative test samples to test set
X_test = []
Y_test = []

for _, row in positive_test_df.iterrows():
    X_test.append(
        np.concatenate((model.wv[row['drug_id']].reshape((1, 650)),
                        model.wv[row['ind_id']].reshape((1, 650))), axis=None)
    )
    Y_test.append(1)

for _, row in negative_test_df.iterrows():
    X_test.append(
        np.concatenate((model.wv[row['drug_id']].reshape((1, 650)),
                        model.wv[row['ind_id']].reshape((1, 650))), axis=None)
    )
    Y_test.append(0)

np.save("./output/X_test.npy", X_test)
np.save("./output/Y_test.npy", Y_test)

# Final Message
print("All tasks completed successfully!")
