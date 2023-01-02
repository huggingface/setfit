clear;

export num_shot_tfew=0
export num_shot_setfit=0

export datasets=(emotion)
export num_iterations=4
export num_unlabeled=30
export top_n_list=(200) # 1000 5000 10000


if [[ $num_shot_tfew -eq 0 ]]
then
  export tfew_train_steps=0
else
  export tfew_train_steps=2000
fi

for dataset in ${datasets[@]}
do
    for train_split in 0 # 1 2 3 4
    do
        # T-Few
        python -m src.pl_train -c t03b.json+ia3.json+${dataset}_pairs.json \
        -k load_weight="t-few/pretrained_checkpoints/t03b_ia3_finish.pt" \
        exp_name=t03b_pretrained/${dataset}/train-${num_shot_tfew}-${train_split}/seed0 \
        train_split=${train_split} \
        few_shot_random_seed=0 \
        seed=0 \
        num_shot=1 \
        batch_size=8 \
        eval_batch_size=16 \
        grad_accum_factor=1 \
        eval_before_training=0 \
        allow_skip_exp=0 \
        unlabeled_iterations=${num_iterations} \
        unlabeled_examples=${num_unlabeled} \
        num_steps=${tfew_train_steps}
      
    # for top_n in ${top_n_list[@]}
    # do
    #     # SetFit
    #     python ../setfit/run_pseudolabeled.py --sample_size=${num_shot_setfit} --dataset=${dataset} --override_results --train_split=${train_split} \
    #             --pseudolabels_path="${num_shot_tfew}_shot/${num_unlabeled}_unlabeled/${num_iterations}_iterations/top_${top_n}/split_${train_split}.json" \
    #             --exp_name=f"tfew_${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}" # --add_data_augmentation
    # done
  done
  # python ../create_summary_table.py --path ../setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-ftfew_${num_shot_tfew}_shot_${num_unlabeled}_unlabeled_${num_iterations}_iter_top_${top_n}
done

