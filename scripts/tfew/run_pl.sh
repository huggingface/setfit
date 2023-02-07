export num_shot=8
export num_shot_pl=8

# export datasets=(SentEval-CR emotion amazon_counterfactual_en enron_spam sst5)
# export num_iterations=20
# export num_unlabeled=2900 #00
# export top_n_list=(200 800 1000 3000 5000) #(200 600 800 1000 3000 5000)
# export splits=(0 1 2)
# export seeds=(1 2 3)
# export setfit_data_aug=0
# export setfit_iter_list=(20) #(0 5 10 13 15 17 20)

# DEBUG
export datasets=(SentEval-CR)
export num_iterations=20
export num_unlabeled=300
export top_n_list=(200) #(200 600 800 1000 3000 5000)
export splits=(1 2)
export seeds=(1)
export setfit_data_aug=0
export setfit_iter_list=(20) #(0 5 10 13 15 17 20)


if [[ $num_shot -eq 0 ]]
then
  export tfew_train_steps=0
  export tfew_splits=(0)
  export tfew_num_shot=1
else
  export tfew_train_steps=2000
  export tfew_splits=${splits}
  export tfew_num_shot=${num_shot}
fi


# Run T-Few
for seed in ${seeds[@]}
do
  for dataset in ${datasets[@]}
  do
    for tfew_split in ${tfew_splits[@]}
    do
      python -m src.pl_train -c t03b.json+ia3.json+pairs.json \
      -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
      dataset=${dataset} \
      pairs=1 \
      exp_name=t03b_pretrained/${dataset}/train-${num_shot}-shot-seed-${seed}/split_${tfew_split} \
      train_split=${tfew_split} \
      few_shot_random_seed=${seed} \
      seed=${seed} \
      num_shot=${tfew_num_shot} \
      batch_size=8 \
      eval_batch_size=128 \
      grad_accum_factor=1 \
      eval_before_training=0 \
      allow_skip_exp=0 \
      unlabeled_iterations=${num_iterations} \
      unlabeled_examples=${num_unlabeled} \
      num_steps=${tfew_train_steps} \
      train_template_idx=${seed}
    done
  done
done

# for seed in ${seeds[@]}
# do
#   for dataset in ${datasets[@]}
#   do
#     for setfit_iter in ${setfit_iter_list[@]}
#     do
#       export exp_name="num_shot=${num_shot}_ex=${num_unlabeled}x${num_iterations}_seed=${seed}_s-iter=${setfit_iter}"
#       for top_n in ${top_n_list[@]}
#       do
#         for train_split in ${splits[@]}
#         do
       
#           if [[ $num_shot_pl -eq 0 ]]
#           then
#             export pl_split=0
#           else
#             export pl_split=${train_split}
#           fi
#           export pseudolabels_path="${dataset}/${num_shot_pl}_shot/seed_${seed}/${num_unlabeled}_unlabeled/${num_iterations}_iterations/split_${pl_split}_pseudolabeled.jsonl"
       
#           python ../setfit/run_pseudolabeled.py \
#                   --sample_size=${num_shot} \
#                   --dataset=${dataset} \
#                   --train_split=${train_split} \
#                   --pseudolabels_path=${pseudolabels_path} \
#                   --num_iterations=${setfit_iter} \
#                   --exp_name=${exp_name}_top_${top_n} \
#                   --top_n=${top_n}
       
#           if [[ $setfit_data_aug -eq 1 ]]
#           then
#             python ../setfit/run_pseudolabeled.py \
#                     --sample_size=${num_shot} \
#                     --dataset=${dataset} \
#                     --train_split=${train_split} \
#                     --pseudolabels_path=${pseudolabels_path} \
#                     --exp_name="${exp_name}_top_${top_n}_aug_warmup" \
#                     --add_data_augmentation \
#                     --top_n=${top_n}
#           fi
#         done # done train_splits
       
#         # Create summary table of all splits
#         python ../create_summary_table.py --path ../setfit/results/${exp_name}_top_${top_n}
#         if [[ $setfit_data_aug -eq 1 ]]
#         then
#           python ../create_summary_table.py --path ../setfit/results/${exp_name}_top_${top_n}_aug_warmup
#         fi
       
#       done # end top_n
#     done # end setfit_iter
#   done # end datasets
# done # end seeds