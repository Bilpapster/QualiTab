DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables WHERE tablename = 'experiments') THEN
        CREATE TABLE experiments (
            experiment_id UUID PRIMARY KEY,
            dataset_name VARCHAR(255) NOT NULL,
            tag VARCHAR(255) NOT NULL,
            start_time NUMERIC NOT NULL,
            end_time NUMERIC NOT NULL,
            n_train INTEGER NOT NULL,
            n_test INTEGER NOT NULL,
            used_default_split BOOLEAN NOT NULL,
            random_seed INTEGER,
            embeddings JSONB
        );
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables WHERE tablename = 'embeddings_experiments') THEN
        CREATE TABLE embeddings_experiments (
            experiment_id UUID PRIMARY KEY,
            dataset_name VARCHAR(255) NOT NULL,
            train_size INTEGER NOT NULL,
            test_size INTEGER NOT NULL,
            number_of_classes INTEGER NOT NULL,
            number_of_features INTEGER NOT NULL,
            random_seed INTEGER,
            test_embeddings JSONB,
            error_type VARCHAR(255),
            corrupted_columns JSONB,
            corrupted_rows JSONB,
            execution_time DOUBLE PRECISION,
            tag VARCHAR(255),
            row_corruption_percent SMALLINT,
            column_corruption_percent SMALLINT,
            gpu_info JSONB
    );
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables WHERE tablename = 'classification_experiments') THEN
        CREATE TABLE classification_experiments (
            experiment_id UUID PRIMARY KEY,
            dataset_name VARCHAR(255) NOT NULL,
            train_size INTEGER NOT NULL,
            test_size INTEGER NOT NULL,
            used_default_split BOOLEAN,
            random_seed INTEGER,
            roc_auc DOUBLE PRECISION,
            accuracy DOUBLE PRECISION,
            recall DOUBLE PRECISION,
            precision DOUBLE PRECISION,
            f1_score DOUBLE PRECISION,
            execution_time DOUBLE PRECISION,
            tag VARCHAR(255)
    );
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables WHERE tablename = 'embedding_evaluation_metrics') THEN
        CREATE TABLE embedding_evaluation_metrics (
            evaluation_id UUID PRIMARY KEY,
            experiment_id UUID REFERENCES embeddings_experiments(experiment_id),
            evaluation_type VARCHAR(255) NOT NULL, -- e.g., 'linear probing', 'clustering'
            metric_name VARCHAR(255) NOT NULL,     -- e.g., 'ROC AUC', 'Purity', 'Avg Cosine Similarity (k=5)'
            metric_value DOUBLE PRECISION NOT NULL,
            weights JSONB,                         -- Optional: to store classifier weights or cluster centroids
            other_info JSONB,                      -- Optional: for any other relevant information
            evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    END IF;

    CREATE INDEX IF NOT EXISTS idx_experiment_id ON embedding_evaluation_metrics (experiment_id);
    CREATE INDEX IF NOT EXISTS idx_evaluation_type ON embedding_evaluation_metrics (evaluation_type);
    CREATE INDEX IF NOT EXISTS idx_metric_name ON embedding_evaluation_metrics (metric_name);
END $$;
