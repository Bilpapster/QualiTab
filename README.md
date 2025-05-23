# QualiTab
*The first open-source framework for **assessing the impact of imperfect data** on tabular foundation models!*


![QualiTab overview](https://github.com/user-attachments/assets/3a819135-12d3-411c-8874-f05664d752b3)

## About

Deep learning models are typically evaluated on high-quality, curated benchmark datasets. But how realistic is this? Real-world data is often far from curated, benchmark ones. In fact, real-world data is often imperfect, containing various types of data quality issues such as missing values, outliers, and noise. These issues can significantly impact the performance of deep learning models, leading to suboptimal results in real-world applications.

**QualiTab** is an open-source experimentation framework specially designed for assessing the impact of data quality issues, especially on **Tabular** Foundation models, a new and exciting area of research. Just like LLMs understand natural language, Tabular Foundation Models are designed to understand tabular data. These models have shown great promise in various applications, with the greatest example being [TabPFN], (https://github.com/PriorLabs/TabPFN) [published on Nature](https://www.nature.com/articles/s41586-024-08328-6)!

*But how are real-world, imperfect data impacting these models?*

If you are interested in this question, you are in the right place! QualiTab is designed to help researchers and practitioners assess the impact of data quality issues on tabular foundation models. It provides a comprehensive set of tools and techniques for evaluating, and analyzing not only the mere predictive performance of these models, but also the changes in their internal representations with the presence of data quality issues.

And the most important! **QualiTab** is designed for extensibility. You can easily add new data quality issues, new tabular foundation models, and new evaluation metrics. This makes it a powerful tool for researchers and practitioners who want to explore the impact of data quality issues on tabular foundation models.

## How to use QualiTab (Documentation in progress)

Straightforward guidelines for making **QualiTab** your very own are on their way! We thank you for your patience while we work on this.

## Acknowledgements

At the current time, **QualiTab** builds upon significant previous contributions in the areas of benchmarking and realistic data generation. We therefore acknowledge the following projects:
- [The OpenML project](https://www.openml.org/) for providing a large collection of datasets and tasks for benchmarking machine learning algorithms.
- [Jenga](https://github.com/schelterlabs/jenga) for providing a framework for generating realistic data quality issues.

## Contact 

If you have any questions, suggestions, or feedback, please feel free to reach out to us. We are always happy to help and collaborate with the community.

