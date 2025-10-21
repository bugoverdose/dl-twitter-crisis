# DL Twitter Crisis
- [Notion](https://www.notion.so/Deep-Learning-CS-7643-24aeb70f53958059aac8dff148295ab5)

# Instructions
1. create a new conda env: `conda env create -f environment.yml`
2. conda activate crisisbench-dl
3. `python train.py`


# Notes about Dataset
- `*_informativeness.tsv` has binary labels informative/not-informative
- `*_humanitarian.tsv` has multi-class labels
- can probably just use `*_humanitarian.tsv` files inside `all_data_en` since we can just focus on English. Datasets within the folder are also already split into train/val/test sets
    - this is also the dataset the authors of the CrisisBench paper use in Section 4: Experiments


