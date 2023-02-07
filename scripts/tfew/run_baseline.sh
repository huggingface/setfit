clear;

export num_shot_setfit=0
export datasets=(sst5 amazon_counterfactual_en enron_spam SentEval-CR)

export zeroshot_splits=(0 1 2)
export num_splits_fewshot=3
export exp_name="${num_shot_setfit}_shot_setfit_baseline"

for dataset in ${datasets[@]}
do
  if [[ $num_shot_setfit -eq 0 ]]
  then
    for train_split in ${zeroshot_splits[@]}
    do
      # SetFit Zero-Shot using Category Names
      python ../setfit/run_pseudolabeled.py \
          --sample_size=0 \
          --dataset=${dataset} \
          --train_split=${train_split} \
          --exp_name=${exp_name} \
          --add_data_augmentation
    done
  else
    # SetFit Few-Shot
    python ../setfit/run_fewshot.py \
        --sample_sizes=${num_shot_setfit} \
        --dataset=${dataset} \
        --num_splits_fewshot=${num_splits_fewshot} \
        --exp_name=${exp_name} \
        --is_test_set=true
  fi
done

python ../create_summary_table.py --path ../setfit/results/${exp_name}