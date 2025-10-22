# DL Twitter Crisis
- [Notion](https://www.notion.so/Deep-Learning-CS-7643-24aeb70f53958059aac8dff148295ab5)

# Instructions
1. create a new conda env: `conda env create -f environment.yml`
2. conda activate crisisbench-dl
3. `python train.py`

# Notes about Dataset

## CrisisBench

- [CrisisNLP - CrisisBench](https://crisisnlp.qcri.org/crisis_datasets_benchmarks) v1.0: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing
  - The crisis benchmark dataset consists data from several different data sources such as CrisisLex (CrisisLex26, CrisisLex6), CrisisNLP, SWDM2013, ISCRAM13, Disaster Response Data (DRD), Disasters on Social Media (DSM), CrisisMMD and data from AIDR. The purpose of this work was to map the class label, remove duplicates and provide a benchmark results for the community.
  - [Github Source code](https://github.com/firojalam/crisis_datasets_benchmarks)
  - Reference: [Alam et al. (2021)](https://arxiv.org/abs/2004.06774)

### Directory Structure

- `data/all_data_en`: all combined english dataset used for the experiments
- `data/individual_data_en`: consists of data used for the experiments as individual data source such as crisisnlp and crisislex
- `data/event_aware_en`: all combined english dataset with event tag (fire, earthquake, flood, ...) are tagged
- `data/data_split_all_lang`: all combined dataset with their train/dev and test splits
- `data/initial_filtering`: all combined dataset duplicate removed data
- `data/class_label_mapped`: all combined dataset initial set of dataset where class label mapped

### Dataset types

- `*_informativeness.tsv` has binary labels informative/not-informative
- `*_humanitarian.tsv` has multi-class labels
  - `infrastructure_and_utilities_damage`
  - `injured_or_dead_people`
  - `sympathy_and_support`
  - `not_humanitarian`
  - `donation_and_volunteering`
  - `caution_and_advice`
  - `response_efforts`, etc

- can probably just use `*_humanitarian.tsv` files inside `all_data_en` since we can just focus on English. Datasets within the folder are also already split into train/val/test sets
  - this is also the dataset the authors of the CrisisBench paper use in Section 4: Experiments (Alam et al., 2021).

# References

- [Alam et al. (2021). CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing](https://arxiv.org/abs/2004.06774)
