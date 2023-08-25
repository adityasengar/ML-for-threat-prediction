import argparse
import sklearn.metrics as sm
from src.data_processing import load_data, generate_scores, perform_pca, prepare_datasets
from src.model import create_model

def main():
    """Main function to run the threat prediction workflow."""
    parser = argparse.ArgumentParser(description="ML-based Threat Prediction")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help="Mode to run: 'train' or 'predict'")
    parser.add_argument('--data_path', type=str, default='data/dataset.csv', help="Path to the dataset CSV file")
    # In a real scenario, you'd also have args for model path, output path, etc.
    
    args = parser.parse_args()

    print("--- Loading and Preparing Data ---")
    df = load_data(args.data_path)
    df_scored = generate_scores(df)
    
    print("\n--- Performing PCA ---")
    X_pca = perform_pca(df_scored)
    
    X_train, X_test, y_train_scaled, y_test, y_scaler = prepare_datasets(df_scored, X_pca)
    
    if args.mode == 'train':
        model = create_model(input_dim=X_train.shape[1])
        print("\n--- Training Model ---")
        model.fit(X_train, y_train_scaled, epochs=10, batch_size=256, validation_split=0.15, verbose=2)
        # In the next step, we will save this model
        
        # Evaluate on the test set after training
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        print("\n--- Evaluation Metrics (Post-Training) ---")
        print(f"Mean absolute error = {sm.mean_absolute_error(y_test, y_pred):.2f}")
        print(f"Mean squared error = {sm.mean_squared_error(y_test, y_pred):.2f}")
        print(f"R2 score = {sm.r2_score(y_test, y_pred):.2f}")

    elif args.mode == 'predict':
        print("\n--- Prediction Mode ---")
        print("This mode will be fully implemented in the next step when model persistence is added.")
        print("For now, it demonstrates the CLI argument parsing.")
        # Example: loading a model and predicting
        # model = load_model('path/to/model.h5')
        # y_pred = model.predict(X_test)
        pass

if __name__ == "__main__":
    main()