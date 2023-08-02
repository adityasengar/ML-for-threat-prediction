from src.data_processing import load_data, generate_scores, perform_pca, prepare_datasets
from src.model import create_model
import sklearn.metrics as sm

def main():
    """Main function to run the threat prediction workflow."""
    print("--- Loading and Preparing Data ---")
    df = load_data('data/dataset.csv')
    df_scored = generate_scores(df)
    
    print("\n--- Performing PCA ---")
    X_pca = perform_pca(df_scored)
    
    X_train, X_test, y_train_scaled, y_test, y_scaler = prepare_datasets(df_scored, X_pca)
    
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

