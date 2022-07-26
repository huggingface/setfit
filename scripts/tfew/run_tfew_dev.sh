for dataset in emotion ag_news
do
    for train_split in 0 1 2 3 4 5 6 7 8 9
    do
        for seed in 0 1 2 3 4
        do
            for sample_size in 8 16 32
            do
                python -m src.pl_train -c t011b.json+ia3.json+${dataset}.json \
                -k load_weight="t-few/pretrained_checkpoints/t011b_ia3_finish.pt" \
                exp_name=t011b_pretrained_${dataset}/train-${sample_size}-${split}_seed${seed} \
                train_split=${train_split} \
                few_shot_random_seed=${seed} \
                seed=${seed} \
                num_shot=$sample_size \
                eval_epoch_interval=50 \
                batch_size=1 \
                eval_batch_size=2 \
                grad_accum_factor=8
            done
        done
    done
done