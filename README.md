# yourcolo

## dataset

The data required for reasoning to fine-tune and our statistics on the dataset are available at the link: https://pan.baidu.com/s/1zjLLsUCy8nmh_kfEXqywhQ. 

Extract password: m6dx.

### project frequent groups file:
k9mail_frequent_groups.json

AntennaPod_frequent_groups.json

cgeo_frequent_groups.json

anki_frequent_groups.json

termux_frequent_groups.json

## fine-tune
--project_name=anki

bash train.sh

## inference
--project_name=anki

bash test.sh
