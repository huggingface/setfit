for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        python t-few/src/pl_train -c t03b.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" exp_name=$EXPERIMENT_NAME few_shot_random_seed=${seed} seed=${seed}
    done
done
