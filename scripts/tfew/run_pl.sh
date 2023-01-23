clear;

export num_shot_tfew=0
export num_shot_setfit=0

export datasets=(sst5 emotion amazon_counterfactual_en enron_spam SentEval-CR)
export num_iterations=10
export num_unlabeled=5000 #1000
export top_n_list=(200 1000)
export splits=(0 1)
export seeds=(0 1)
export setfit_data_aug=1

## DEBUG
# export num_iterations=4
# export num_unlabeled=50
# export splits=(0)
# export seeds=(0)
# export setfit_data_aug=0
# export top_n_list=(200)


if [[ $num_shot_tfew -eq 0 ]]
then
  export tfew_train_steps=0
else
  export tfew_train_steps=2000
fi

for dataset in ${datasets[@]}
do
  for seed in ${seeds[@]}
  do
      for train_split in ${splits[@]}
      do
          python -m src.pl_train -c t03b.json+ia3.json+pairs.json \
          -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
          dataset=${dataset} \
          pairs=1 \
          exp_name=t03b_pretrained/${dataset}/train-${num_shot_tfew}-${train_split}/seed0 \
          train_split=${train_split} \
          few_shot_random_seed=${seed} \
          seed=${seed} \
          num_shot=1 \
          batch_size=8 \
          eval_batch_size=16 \
          grad_accum_factor=1 \
          eval_before_training=0 \
          allow_skip_exp=0 \
          unlabeled_iterations=${num_iterations} \
          unlabeled_examples=${num_unlabeled} \
          num_steps=${tfew_train_steps}

        # SetFit Zero-Shot using Category Names
        # python ../setfit/run_pseudolabeled.py --sample_size=${num_shot_setfit} --dataset=${dataset} --train_split=0 \
        #             --exp_name=f"${num_shot_tfew}_shot_seed_${seed}_aug" --add_data_augmentation

        for top_n in ${top_n_list[@]}
        do
            python ../setfit/run_pseudolabeled.py --sample_size=${num_shot_setfit} --dataset=${dataset} --train_split=${train_split} \
                    --pseudolabels_path="${dataset}/${num_shot_tfew}_shot/seed_${seed}/${num_unlabeled}_unlabeled/${num_iterations}_iterations/split_${train_split}_pseudolabeled.jsonl" \
                    --exp_name="${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}_seed_${seed}" \
                    --top_n=${top_n}
            python ../create_summary_table.py --path ../setfit/results/${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}_seed_${seed}

            if [[ $setfit_data_aug -eq 1 ]]
            then
              python ../setfit/run_pseudolabeled.py --sample_size=${num_shot_setfit} --dataset=${dataset} --train_split=${train_split} \
                      --pseudolabels_path="${dataset}/${num_shot_tfew}_shot/seed_${seed}/${num_unlabeled}_unlabeled/${num_iterations}_iterations/split_${train_split}_pseudolabeled.jsonl" \
                      --exp_name="${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}_seed_${seed}_pl_aug" --add_data_augmentation \
                      --top_n=${top_n}
              python ../create_summary_table.py --path ../setfit/results/${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}_seed_${seed}_pl_aug
            fi
        done
    done
  done
done
