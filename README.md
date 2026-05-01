<div align="left">
    <img src="quickstart/images/logo.png"
         alt="Image of Fiddler logo" width="200"/>
</div>

***

This repo contains example notebooks and accompanying documentation for using Fiddler.

# Examples

These example notebooks aim to give you a quick start on various Fiddler capabilities using different model tasks, data types, and use cases. They can also serve as a reference guide for setting up the monitoring of your own models in Fiddler.

## Getting Started

Use the projects in this repo to onboard models and data to illustrate ML model and LLM application monitoring, analysis, and protection.
This repo contains the example notebooks listed below. You can launch them in a Google Colab environment using the Colab links.

> **Note:** This repository uses Git Large File Storage (Git LFS) for managing large files. 
> Please make sure you have Git LFS installed before cloning this repository. 
> You can find installation instructions at [git-lfs.github.com](https://git-lfs.github.com/).
> You can find the file types tracked via GitLFS at the `.gitattributes` file (currently only `.csv` files)

## The Basics

* [LLM - Comparison](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_LLM_Comparison.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_LLM_Comparison.ipynb)
* [LLM - Simple Monitoring Quickstart](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_LLM_Chatbot.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_LLM_Chatbot.ipynb)
* [ML - Simple Monitoring Quickstart](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Simple_Monitoring.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Simple_Monitoring.ipynb)
* [Managing Model Versions with Fiddler](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Model_Versions.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Model_Versions.ipynb)
* [User-defined Feature Impact Upload](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_User_Defined_Feature_Impact.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_User_Defined_Feature_Impact.ipynb)

## Specific Use Cases

* [Image/Computer Vision Model Monitoring ](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Image_Monitoring.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Image_Monitoring.ipynb)
* [NLP Model Monitoring - Multiclass Classification](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_NLP_Multiclass_Monitoring.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_NLP_Multiclass_Monitoring.ipynb)
* [Class Imbalance Drift Detection](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Imbalanced_Data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Imbalanced_Data.ipynb)
* [Ranking Model - Monitoring](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Ranking_Model.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Ranking_Model.ipynb)
* [Regression Model - Monitoring](https://github.com/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Regression_Model.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/quickstart/latest/Fiddler_Quickstart_Regression_Model.ipynb)

## Cookbooks

Use-case oriented notebooks that demonstrate end-to-end workflows for solving real AI evaluation and monitoring problems with Fiddler. Each cookbook has a companion guide on the [Fiddler docs site](https://docs.fiddler.ai/developers/cookbooks).

* [RAG Evaluation Fundamentals](https://github.com/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_RAG_Evaluation_Fundamentals.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_RAG_Evaluation_Fundamentals.ipynb) — Evaluate RAG quality with built-in evaluators
* [RAG Experiments at Scale](https://github.com/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_RAG_Experiments_at_Scale.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_RAG_Experiments_at_Scale.ipynb) — Compare RAG pipeline configurations systematically
* [Custom Judge Evaluators](https://github.com/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_Custom_Judge_Evaluators.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiddler-labs/fiddler-examples/blob/main/cookbooks/Fiddler_Cookbook_Custom_Judge_Evaluators.ipynb) — Create domain-specific evaluation criteria

## Fiddler Utils — Admin Automation Library

`fiddler-utils` is an admin automation library for Fiddler that provides
high-level abstractions for common administrative tasks (connection
management, asset export/import, FQL parsing, schema validation, model
comparison, and more). It is **not part of the official Fiddler SDK**, but
both Fiddler field engineers and customers are welcome to use and extend it.

The library now lives in its own public repository:
**https://github.com/fiddler-labs/fiddler-utils**

Install the latest tagged release from git:

```bash
pip install "fiddler-utils @ git+https://github.com/fiddler-labs/fiddler-utils.git@v1.0.1"
# or with uv:
uv add "git+https://github.com/fiddler-labs/fiddler-utils.git@v1.0.1"
```

For the OpenTelemetry-based event ingestion utilities
(`log_pandas_traces`, `log_event_traces`), install with the `[otel]` extra:

```bash
pip install "fiddler-utils[otel] @ git+https://github.com/fiddler-labs/fiddler-utils.git@v1.0.1"
```

Quick example:

```python
from fiddler_utils import get_or_init, ModelComparator, SegmentManager

get_or_init(url="https://your-org.cloud.fiddler.ai", token="...")
```

See the [fiddler-utils README](https://github.com/fiddler-labs/fiddler-utils#readme)
for the full API reference and usage examples.

Several notebooks in the [`misc-utils`](./misc-utils) directory demonstrate `fiddler-utils` features:

* **[export_import_models.ipynb](./misc-utils/export_import_models.ipynb)** - Complete model export/import workflows
* **[export_import_model_assets.ipynb](./misc-utils/export_import_model_assets.ipynb)** - Asset-level transfers between models
* **[compare_models.ipynb](./misc-utils/compare_models.ipynb)** - Comprehensive model comparison
* **[env_stats.ipynb](./misc-utils/env_stats.ipynb)** - Environment reporting and analysis

## Miscellaneous Utilities

The [misc-utils](./misc-utils) directory contains utility notebooks for customer success engineers, field AI engineers, and solution engineers. These notebooks provide tools for various administrative tasks and solutions to common challenges when working with Fiddler deployments. See the [misc-utils README](./misc-utils/README.md) for a detailed catalog of available utilities.

## License

This project is licensed under the MIT license. See the [LICENSE](https://github.com/fiddler-labs/fiddler-examples/blob/main/LICENSE) file for more info.

## Additional Resources

* [Documentation](https://docs.fiddler.ai)
* [Fiddler Blog](https://www.fiddler.ai/blog)
* [Fiddler Resource Library](https://www.fiddler.ai/resources)
