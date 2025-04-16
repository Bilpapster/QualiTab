-- init.sql
-- Check if the table exists
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
END $$;