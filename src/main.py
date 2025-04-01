from load_cleanML_dataset import load_dataset

if __name__ == "__main__":
    dataset_name = 'Credit'

    # Example 1: No target column specified
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, target_column='SeriousDlqin2yrs')
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
