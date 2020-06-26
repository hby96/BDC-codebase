# Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/hby96/BDC-codebase.git
   cd BDC-codebase
   ```

   

# Getting Started

## save order csv file to feature csv file

1. modify the folder path in the `save_csv_to_res_feature.py`.

2. run the following command:

   ```shell
   python3 lib/utils/save_csv_to_res_feature.py
   ```



## train model

1. modify the configurations in the config file.

2. run the followin command:

   ```shell
   python3 main/train_res.py --cfg configs/load_res_feature.yaml
   ```

   

