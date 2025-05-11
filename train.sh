lang=java
project_name=anki
mkdir -p ./saved_models/$project_name
python run.py \
    --output_dir=./saved_models/$project_name \
    --model_name_or_path=/home/chikuo/pretrain_model/model_repo/unixcoder/ \
    --do_train \
    --train_data_file=data/$project_name/all_train_add_intro.json \
    --eval_data_file=data/$project_name/${project_name}_valid.json \
    --codebase_file=data/$project_name/${project_name}_code.json \
    --num_train_epochs 30 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --project_name $project_name
    --learning_rate 2e-5 \
    --result_file saved_models/${project_name}_result_unixcoder.json \
    --seed 1234567 2>&1| tee saved_models/train_${project_name}.log 

