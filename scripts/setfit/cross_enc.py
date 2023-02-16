from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CECorrelationEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
import numpy as np 
from scipy.special import softmax
from sentence_transformers import InputExample


def train_cross_enc(cross_enc_model,
                            train_data, 
                            batch_size, 
                            num_epochs, 
                            num_labels, 
                            unlabeled_data_dict):

 # load our training data (first 95%) into a dataloader
    loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )

    cross_encoder = CrossEncoder(cross_enc_model, num_labels=num_labels)

    # warmup is minimum 10 steps (10 applied in low few shot)
    warmup = max(int(len(loader) * num_epochs * 0.1),int(10))
    #warmup = int(len(loader) * num_epochs * 0.1)

    cross_encoder.fit(
        train_dataloader=loader,
        epochs=num_epochs,
        warmup_steps=warmup
    )

    #************ Cross-encoder evaluation ************
    
    test_sentence_pairs = unlabeled_data_dict['sentence_pairs']
    test_gold_labels = unlabeled_data_dict['gold_labels']
    pred_scores = cross_encoder.predict(test_sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
    probabilities = softmax(pred_scores, axis=1)
    pred_labels = np.argmax(pred_scores, axis=1)
    # probability of winning class (0 or 1)
    wining_class_prob = np.max(probabilities, axis=1)
    
    # sort data according to probability
    sort_idx = sorted(range(len(wining_class_prob)), key=wining_class_prob.__getitem__, reverse=True)
    unlabeled_sentence_pairs_sorted = [test_sentence_pairs[i] for i in sort_idx]
    unlabeled_gold_labels_sorted = [test_gold_labels[i] for i in sort_idx]

    unlabeled_data_dict.update({"pred_scores": pred_scores, 'pred_labels': pred_labels, 'unlabeled_sentence_pairs_sorted': unlabeled_sentence_pairs_sorted, 'unlabeled_gold_labels_sorted': unlabeled_gold_labels_sorted })

    return cross_encoder, unlabeled_data_dict


def evaluate_cross_encoder(cross_encoder, test_sentence_pairs, test_gold_labels):
    #test_sentence_pairs = test_data_dict['sentence_pairs']
    #gold_labels = test_data_dict['gold_labels']
    test_data = []
    for pair, gold_label in zip(test_sentence_pairs, test_gold_labels):
        test_data.append(InputExample(texts=pair, label=gold_label))

    # eval for full test set
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test_data)
    pred_result = round(evaluator(cross_encoder), 3)

    return pred_result

   
def generate_unified_format_dataset(in_data, sentence1, sentence2, label):
    data = []
    for row in in_data:
        data.append(
            InputExample(
                #texts=[row[sentence1]] if task == "sst2" or task == "cola" else [row[sentence1], row[sentence2]],
                texts= [row[sentence1], row[sentence2]],
                label= row[label]
            )
        )
    return data 