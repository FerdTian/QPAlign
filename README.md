# Experiment Instructions

This repository contains the code and scripts used for the experiments in our paper. Below are the steps to install dependencies and reproduce the results on different datasets and simulated graph structures.

---

## 1. Installation

Before running the experiments, please install all required Python packages:

```bash
pip install -r requirements.txt
```

This will ensure that the environment is configured with the proper versions of libraries used in our experiments.

---

## 2. Running Experiments on Real Datasets

We provide scripts to reproduce results on three benchmark datasets. Each script will automatically set the necessary parameters and start the training/evaluation process.

* **ACM-DBLP dataset**
  Run the following script to reproduce the experiment on the ACM-DBLP citation network:

```bash
bash scripts/run_acmdblp.sh
```

* **Douban dataset**
  To reproduce results on the Douban social network dataset:

```bash
bash scripts/run_douban.sh
```

* **Cancer slice dataset**
  For experiments on the biological dataset (cancer slices):

```bash
bash scripts/run_bio.sh
```

---

## 3. Running Simulations on Synthetic Graphs

In addition to real datasets, we include experiments on simulated graphs to evaluate the model’s performance under controlled synthetic settings.

* **Gaussian graph simulation**
  This script generates a Gaussian graph and runs the corresponding experiment:

```bash
bash scripts/run_gaussian.sh
```

* **Erdős–Rényi (ER) graph simulation**
  This script simulates an ER random graph and runs the experiment:

```bash
bash scripts/run_er.sh
```

---

## Notes

* All scripts can be executed directly from the root directory of the project.
* The scripts include pre-set hyperparameters used in the paper. If you wish to modify them, you can edit the corresponding script file in `scripts/`.
* Results will be saved in the designated output directory specified within each script.
