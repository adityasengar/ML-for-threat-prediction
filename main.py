import argparse
import os
import sklearn.metrics as sm
from keras.models import load_model
from src.data_processing import load_data, generate_scores, perform_pca, prepare_datasets
from src.model import create_model

def main():
    """Main function to run the threat prediction workflow."""
    parser = argparse.ArgumentParser(description="ML-based Threat Prediction")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help="Mode to run: 'train' or 'predict'")
    parser.add_argument('--data_path', type=str, default='data/dataset.csv', help="Path to the dataset CSV file")
    parser.add_argument('--model_path', type=str, default='threat_model.h5', help="Path to save or load the model")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Loading and Preparing Data for Training ---")
        df = load_data(args.data_path)
        df_scored = generate_scores(df)
        
        print("\n--- Performing PCA ---")
        X_pca = perform_pca(df_scored)
        
        X_train, X_test, y_train_scaled, y_test, y_scaler = prepare_datasets(df_scored, X_pca)
        
        model = create_model(input_dim=X_train.shape[1])
        print("\n--- Training Model ---")
        model.fit(X_train, y_train_scaled, epochs=10, batch_size=256, validation_split=0.15, verbose=2)
        
        print(f"\n--- Saving model to {args.model_path} ---")
        model.save(args.model_path)
        
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        print("\n--- Evaluation Metrics (Post-Training) ---")
        print(f"Mean absolute error = {sm.mean_absolute_error(y_test, y_pred):.2f}")
        print(f"R2 score = {sm.r2_score(y_test, y_pred):.2f}")

    elif args.mode == 'predict':
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}. Please train the model first using --mode train.")
            return

        print(f"--- Loading model from {args.model_path} ---")
        model = load_model(args.model_path)

        print("--- Loading and Preparing Data for Prediction ---")
        df = load_data(args.data_path)
        df_scored = generate_scores(df)
        X_pca = perform_pca(df_scored)
        _, X_test, _, y_test, y_scaler = prepare_datasets(df_scored, X_pca)

        print("\n--- Making Predictions ---")
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        print("\n--- Prediction Evaluation ---")
        print(f"Mean absolute error = {sm.mean_absolute_error(y_test, y_pred):.2f}")
        print(f"R2 score = {sm.r2_score(y_test, y_pred):.2f}")
        # Here you could save the predictions to a file, etc.

if __name__ == "__main__":
    main()
