import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def kdd_data(train_path, test_path):
    # load csvs and combine for consistent encoding
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df,test_df], axis=0)

    # clean label values and binary encoding
    full_df['label'] = full_df['label'].str.strip()
    # if label is not 'normal', set to 0; otherwise, set to 1
    full_df['label'] = full_df['label'].where(full_df['label'] != 'normal', 0)
    full_df['label'] = full_df['label'].where(full_df['label'] == 0,1)

    # columns that show be encoded to numbers
    columns = ['protocol_type', 'service', 'flag',]
    for col in columns:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
    
    #split back into train and test sets
    train_df = full_df.iloc[:len(train_df)]
    test_df = full_df.iloc[len(train_df):]

    # divide features and labels
    x_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    x_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test
