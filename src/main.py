from load_cleanML_dataset import load_dataset

if __name__ == "__main__":
    dataset_name = 'Airbnb'
    train_df, test_df = load_dataset(dataset_name)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")