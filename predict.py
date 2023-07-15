import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
import sklearn.metrics as sm

def load_data(path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)

def perform_pca(df):
    """Performs PCA on the dataframe."""
    features = ['inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)', 'vsession_duration(min)', 'ip_type', 'num_sessions', 'num_users', 'num_unique_apis', 'source']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    
    print("Shape after PCA:", principalComponents.shape)
    return principalComponents

def create_model(input_dim):
    """Creates the DNN model."""
    model = Sequential()
    num = 128
    model.add(Dense(num, activation='relu', input_dim=input_dim))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(num, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main():
    """Main function to run the threat prediction workflow."""
    # Load and prepare data
    df = load_data('data/dataset.csv')

    # Create the 'score' column (simplified from notebook)
    def score_func(value):
        return np.random.uniform(0, 5) if value == 0 else np.random.uniform(5, 10)
    df['score'] = df['behavior_type'].map(score_func)
    
    # PCA
    X_pca = perform_pca(df)
    
    # Prepare training and test data
    f = 0.75
    countrain = int(f * len(df))
    
    # Shuffle data
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    X_pca = X_pca[indices]
    y = df['score'].values[indices]

    X_train, X_test = X_pca[:countrain], X_pca[countrain:]
    y_train, y_test = y[:countrain], y[countrain:]
    
    # Scale target variable
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

    # Create and train the model
    model = create_model(input_dim=X_train.shape[1])
    print("\n--- Training Model ---")
    model.fit(X_train, y_train_scaled, epochs=10, batch_size=256, validation_split=0.15, verbose=2)

    # Evaluate the model
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Mean absolute error = {sm.mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean squared error = {sm.mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 score = {sm.r2_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    main()
