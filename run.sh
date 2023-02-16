#!/bin/bash

##########################

export SEEDS=(1 2) #(5 42 12 27 32) # QUICK: (5 42 12)
export SAMPLE_SIZES=(16) #(10 15 20 25 35 40 45 50 60 70 90 100 200 1000)

export DATASETS=("emotion")
export NUM_ITERATIONS=20
export NUM_EPOCHS=10
export BATCH_SIZE=16
export MODEL=paraphrase-mpnet-base-v2
export CROSS_ENC_MODEL=bert-base-uncased
export LABELED_SPLIT_SIZE=5000
export UNLABELED_SPLIT_SIZE=2000

############################

export DEVICE=gpu

if [[ $DEVICE == cpu ]]
then
   export CUDA_VISIBLE_DEVICES="-1";
else
   export CUDA_VISIBLE_DEVICES="2";
fi

export TOKENIZERS_PARALLELISM=false
export OUTPUT=models/"$MODEL"_bs_"$BS"_NPE_"$NP_EXTRACTORS"

###############################################

# for NP_EXTRACTORS in "${NP_EXTRACTORS_LIST[@]}"
# do
export TIME=$(date +"%m.%d.%H.%M")
# export RESULTS_ROOT=results/"$TIME"_bs="$BS_train"_"$NP_EXTRACTORS"_all_seeds

# for MAX_TRAIN_STEPS in "${MAX_TRAIN_STEPS_LIST[@]}"
# do

export RESULTS_ROOT=results/"$TIME"_"HPARAM_SWEEP$SMOKE_STR" #_NPE="$NP_EXTRACTORS" #_STEPS="$MAX_TRAIN_STEPS"
mkdir -p $RESULTS_ROOT

{ git log --name-status HEAD^..HEAD & pip freeze; } > $RESULTS_ROOT/"env.log"

# for FEW_SHOT in "${MODEL_OR_BASELINE[@]}"
# for BS_train in "${BS_LIST[@]}"


python /home/oren_nlp/setfit/scripts/setfit/pseudo_label.py \
      --model $MODEL \
      --cross_enc_model $CROSS_ENC_MODEL \
      --datasets $DATASETS \
      --num_iterations $NUM_ITERATIONS \
      --num_epoch $NUM_EPOCHS \
      --batch_size $BATCH_SIZE \
      --seed $SEEDS \
      --labeled_split_size $LABELED_SPLIT_SIZE \
      --unlabeled_split_size $UNLABELED_SPLIT_SIZE \


echo ""
echo "EXPERIMENT COMPLETED."