import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load_data(path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)

def generate_scores(df):
    """Creates the 'score' column."""
    def score_func(value):
        return np.random.uniform(0, 5) if value == 0 else np.random.uniform(5, 10)
    df['score'] = df['behavior_type'].map(score_func)
    return df

def perform_pca(df):
    """Performs PCA on the dataframe."""
    features = ['inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)', 'vsession_duration(min)', 'ip_type', 'num_sessions', 'num_users', 'num_unique_apis', 'source']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    
    print("Shape after PCA:", principalComponents.shape)
    return principalComponents

def prepare_datasets(df, X_pca):
    """Splits data into training and testing sets and scales the target."""
    f = 0.75
    countrain = int(f * len(df))
    
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    X_pca = X_pca[indices]
    y = df['score'].values[indices]

    X_train, X_test = X_pca[:countrain], X_pca[countrain:]
    y_train, y_test = y[:countrain], y[countrain:]
    
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    
    return X_train, X_test, y_train_scaled, y_test, y_scaler
