# DL Twitter Crisis
- [Team Notion](https://www.notion.so/Deep-Learning-CS-7643-24aeb70f53958059aac8dff148295ab5)
- [Project Instruction](./paper/Group_Project_Description.pdf)
- [Project Report format](https://www.overleaf.com/project/5f5ec061aa94370001943266)
- Goal: replicate the results from [Katalinić & Dunđer (2025)](https://www.mdpi.com/2079-9292/14/11/2273)([pdf](./paper/Neural%20Network-Based%20Sentiment%20Analysis%20and%20Anomaly%20Detection%20in%20Crisis-Related%20Tweets.pdf))

# Instructions
1. create a new conda env: `conda env create -f environment.yml` 
2. update existing conda env: `conda env update -f environment.yml`
3. conda activate crisisbench-dl

- Google Colab version (Jade): `anomaly_detection.ipynb`
- Local Run version (Jinwoo): `anomaly_detection_local.ipynb`

# Notes about Dataset

## Turkey and Syria Earthquake Tweets

- [Kaggle - Turkey and Syria Earthquake Tweets](https://crisisnlp.qcri.org/crisis_datasets_benchmarks)
  - Reference: [Katalinić & Dunđer (2025)](https://www.mdpi.com/2079-9292/14/11/2273)

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

# Papers

- [Yasin Kabir et al. (2019). A Deep Learning Approach for Tweet Classification and Rescue Scheduling for Effective Disaster Management](https://arxiv.org/abs/1908.01456) : [pdf](./paper/A%20Deep%20Learning%20Approach%20for%20Tweet%20Classification%20and%20Rescue%20Scheduling%20for%20Effective%20Disaster%20Management.pdf)
- [Alam et al. (2021). CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing](https://arxiv.org/abs/2004.06774) : [pdf](./paper/CrisisBench_Benchmarking%20Crisis-related%20Social%20Media%20Datasets%20for%20Humanitarian%20Information%20Processing.pdf)
- [McDaniel et al. (2024). Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models](https://arxiv.org/abs/2410.00182) : [pdf](./paper/Zero-Shot%20Classification%20of%20Crisis%20Tweets%20Using%20Instruction-Finetuned%20Large%20Language%20Models.pdf)
- Main paper: [Katalinić & Dunđer (2025). Neural Network-Based Sentiment Analysis and Anomaly Detection in Crisis-Related Tweets](https://www.mdpi.com/2079-9292/14/11/2273) : [pdf](./paper/Neural%20Network-Based%20Sentiment%20Analysis%20and%20Anomaly%20Detection%20in%20Crisis-Related%20Tweets.pdf)
