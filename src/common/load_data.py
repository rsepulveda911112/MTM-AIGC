import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"                                        
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"                                        
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def load_data(file, is_features):
    df = pd.read_csv(file,sep="\t")
    df["text"] = df.text.apply(preprocess)
    print(df['label'].value_counts())
    # To labels
    if 'label' in df.columns:
        labels = df['label']
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels.values)
    else:
        labels = [] * len(df)

    texts = df['text']
    features = [[]] * len(df)
    if is_features:

        difficult = df['difficult']
        difficultES = df['difficultES']
        understable = df['difficultES']
        neg = df['neg']
        neu = df['neu']
        pos = df['pos']
        compound = df['compound']
        sentiment = df['sentiment']
        label_encoder_1 = LabelEncoder()
        sentiment = label_encoder_1.fit_transform(sentiment.values)
        features = list(zip(list(difficult), list(difficultES), list(understable), list(neg), list(neu), list(pos), list(compound), list(sentiment)))
        df_features = pd.DataFrame(features)
        features = df_features[:].values

    list_of_tuples = list(zip(list(texts), list(labels), list(features)))

    df_result = pd.DataFrame(list_of_tuples, columns=['text', 'labels', 'features'])
    print(df_result['labels'].value_counts())
    return df_result, df["id"]


def load_data_mul(file, is_features, label_values):
    df = pd.read_csv(file,sep="\t")
    df["text"] = df.text.apply(preprocess)
    label_dict = {}
    # To labels
    for label_value in label_values:
        if label_value in df.columns:
            labels = df[label_value]
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels.values)
            print(df[label_value].value_counts())
            label_dict[label_value] = labels
        else:
            labels = [] * len(df)

    texts = df['text']
    features = [[]] * len(df)
    if is_features:

        difficult = df['difficult']
        difficultES = df['difficultES']
        understable = df['difficultES']
        neg = df['neg']
        neu = df['neu']
        pos = df['pos']
        compound = df['compound']
        sentiment = df['sentiment']
        label_encoder_1 = LabelEncoder()
        sentiment = label_encoder_1.fit_transform(sentiment.values)
        features = list(zip(list(difficult), list(difficultES), list(understable), list(neg), list(neu), list(pos), list(compound), list(sentiment)))
        df_features = pd.DataFrame(features)
        features = df_features[:].values

    list_of_tuples = list(zip(list(texts), list(features)))

    df_result = pd.DataFrame(list_of_tuples, columns=['text', 'features'])
    df_label = pd.DataFrame(label_dict)
    df_result = pd.concat([df_result, df_label], axis=1)
    return df_result, df["id"]
