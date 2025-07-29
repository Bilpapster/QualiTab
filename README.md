# QualiTab
*The first open-source framework for **assessing the impact of imperfect data** on tabular foundation models.*


![QualiTab overview](https://github.com/user-attachments/assets/3a819135-12d3-411c-8874-f05664d752b3)

Official, actively extended codebase for the research paper *Out in the Wild: Investigating the Impact of Imperfect Data on a Tabular Foundation Model*, appearing in the [QDB workshop](https://qdb-workshop.github.io/) of the [VLDB 2025 Conference](https://vldb.org/2025/) (London, UK - September 2025).

```bibtex
@inproceedings{papastergios2025qualitab,
   title={Out in the Wild: Investigating the Impact of Imperfect Data on a Tabular Foundation Model},
   author={Papastergios, Vasileios and Gounaris, Anastasios},
   booktitle={14th International Workshop on Quality in Databases (QDB’25)},
   year={2025},
   month = sep, 
   url={TBA}
}
```
> *Note:* This is an evolving codebase, actively extended to cover more models (e.g., [TabICL](https://github.com/soda-inria/tabicl)), more datasets (e.g., the [TabArena benchmark](https://arxiv.org/abs/2506.16791)) and even more ways to evaluate the impact of imperfect data (e.g., predictive performance and synthetic data generation). If your goal is to reproduce the exact experimental results of our QDB'25 paper, you can use the codebase snapshot identified by commit c5976ac6bf3a67a5fc77b887d9406eee3ebc3f27. Otherwise, our suggestion is to use the latest version to have access to all the latest features and improvements.

## About

Deep learning models are typically evaluated on high-quality, curated benchmark datasets. _But how realistic is this?_ Real-world datasets are often far from curated, benchmark ones. In fact, real-world data is often imperfect, containing various types of data quality issues such as missing values, outliers, and noise. These issues can significantly impact the performance of deep learning models, leading to suboptimal results in real-world applications.

**QualiTab** is an open-source experimentation framework specially designed for assessing the impact of data quality issues, especially on **Tabular** Foundation models, a new and exciting area of research. Just like LLMs understand natural language, Tabular Foundation Models are designed to understand tabular data. These models have shown great promise in various applications, with the greatest example being [TabPFN](https://github.com/PriorLabs/TabPFN), [published on Nature](https://www.nature.com/articles/s41586-024-08328-6)!

---

*But how are real-world, imperfect data impacting these models? What data cleaning strategies would be the most effective when combined with a specific model (e.g., TabPFN) and task at hand (e.g., classification)?*

---

If you are interested in these question, you are in the right place! **QualiTab** is designed to help researchers and practitioners assess the impact of data quality issues on tabular foundation models. It provides a comprehensive set of tools and techniques for evaluating, and analyzing not only the mere predictive performance of these models, but also the changes in their internal representations with the presence of data quality issues.

And the most important! **QualiTab** is designed for extensibility. You can easily add new data quality issues, new tabular foundation models, and new evaluation metrics. This makes it a powerful tool for researchers and practitioners who want to explore the impact of data quality issues on tabular foundation models.

## How to use QualiTab

There are two main ways to use **QualiTab**:
1. You can easily replicate our experiments investigating the impact of data quality issues on the embeddings of [TabPFN](https://github.com/PriorLabs/TabPFN); or
2. You can use **QualiTab** to create your own experiments with your own data quality issues, tabular foundation model(s), and/or datasets. 

Both use cases are described below. The only requirements to use **QualiTab** are Python 3.10+ and [Docker](https://www.docker.com/).

#### How to replicate our experiments

To replicate our experiments on the impact of 3 different data quality issues on TabPFN's embeddings, follow these steps:

1. Clone the repository and get into the root directory of the project:
   ```bash
   git clone https://github.com/Bilpapster/QualiTab.git
   cd QualiTab
    ```
2. Create a`.env` file in the root directory of the project with the following configuration lines:
    ```bash
   echo "
    POSTGRES_USER=username
    POSTGRES_PASSWORD=secret
    POSTGRES_DB=experiments
    POSTGRES_MAPPED_PORT=15432
    POSTGRES_HOST=localhost
    SEEDS=16840,190459,318736,390196,511033,577121,577764,759700,921332,924126
    DATASETS_TO_SKIP=40927,40996,554,23517,38
   " > .env
    ```
   > Note that the `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, and `POSTGRES_MAPPED_PORT` variables can be set to any values you like the first time you run the experiments. After the first run, the database instance will be created with these credentials, so you should consistently use the same values in subsequent runs. The `POSTGRES_MAPPED_PORT` variable should be set to a port that is not in use on your machine, so you can access the database instance from your host machine. The `SEEDS` variable is used to control the random seed for reproducibility, and the `DATASETS_TO_SKIP` variable is used to skip certain datasets that are known to cause issues during the experiments. If you want to exactly replicate our experiments, you should use the same values as in the example above. If exact replication is not required, you can change these values to any other valid ones, but make sure to keep the same format. To make sure that the `.env` file is correctly created with the configurations of your choice you can run `cat .env` to see its contents.
3. Start all services using Docker Compose:
   ```bash
   docker compose up --build -d
   ```
4. Run the script we have used for our experiments:
   ```bash
   docker exec -it tabpfn python main_embeddings.py
   ```
   > **Important**: This can take a significant amount of time (> 45h) to complete, even if running on a GPU. If you are using CPU only, the execution time will be significantly higher. We suggest using a GPU for faster execution. See Kaggle's and Google Colab's free GPU offerings for a quick start. The `main_embeddings.py` script will run the experiments and store the results in the PostgreSQL database instance running in the Docker container. You can run the experiments over multiple runs, the script automatically avoids re-calculating the embeddings for datasets that have already been processed.
5. (Optional) You can see the progress of the experiments by accessing the Adminer web interface at [http://localhost:8080](http://localhost:8080). Use the credentials you set in the `.env` file to log in (make sure to select "Postgres" under the "System" option). You can then browse the `embeddings_experiments` table to see the results of the experiments.

#### How to create your own experiments

Straightforward guidelines for making **QualiTab** your very own are on their way! We thank you for your patience while we work on this.

## Acknowledgements

At the current time, **QualiTab** builds upon significant previous contributions in the areas of benchmarking and realistic data corruption. We therefore acknowledge the following projects:
- [The OpenML project](https://www.openml.org/) for providing a large collection of datasets and tasks for benchmarking machine learning algorithms.
- [Jenga](https://github.com/schelterlabs/jenga) for providing a framework for injecting realistic data quality issues into ML datasets.

## Contact 

If you have any questions, suggestions, or feedback, please feel free to reach out to us. We are always happy to help and collaborate with the community.

