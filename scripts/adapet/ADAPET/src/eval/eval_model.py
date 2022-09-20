import os
import json
import torch
#added
import os
import time 
import numpy as np

from src.eval.Scorer import Scorer
from src.eval.Writer import Writer

def eval(config, model, batch_iter, scorer):
    '''
    Evaluate model

    :param config:
    :param model:
    :param batch_iter:
    :param scorer:
    :return:
    '''
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batch_iter):
            pred_lbl, lbl_logits = model.predict(batch)
            list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"]["idx"].cpu().numpy().tolist()
            list_lbl = batch["output"]["true_lbl"] if "true_lbl" in batch["output"] else batch["output"]["lbl"]

            if config.dataset.lower() == 'fewglue/record':
                true_lbl = torch.tensor([1])
                pred_lbl = torch.tensor([list_lbl[0][pred_lbl[0].item()]])
                scorer.add_batch(list_idx, pred_lbl, true_lbl, lbl_logits.cpu().numpy(), None)
            else:
                scorer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy(), None)



def dev_eval(config, model, batcher, num_batches, dict_avg_val=None):
    '''
    Evaluates the accuracy on the dev partition

    :param config:
    :param model:
    :param batcher: batcher to get batches of data
    :param num_batches:
    :param dict_avg_val: dictionary storing metrics

    :return: currrent dev score
    '''

    dict_eval = {}
    dict_eval["num_batches"] = num_batches

    if dict_avg_val is not None:
        dict_eval.update(dict_avg_val)

    # Get train Score
    if config.eval_train:
        train_scorer = Scorer(config, config.dataset)
        train_iter = batcher.get_eval_train_batch()
        eval(config, model, train_iter, train_scorer)
        _, train_scores = train_scorer.get_score("train")
        dict_eval.update(train_scores)

    # Get dev Score
    if config.eval_dev:
        dev_scorer = Scorer(config, config.dataset)
        dev_iter = batcher.get_dev_batch()
        eval(config, model, dev_iter, dev_scorer)
        score_eval, dev_scores = dev_scorer.get_score("dev")
        dict_eval.update(dev_scores)
        dev_logits = dev_scorer.get_logits()
    else:
        score_eval = 0
        dev_logits = None

    with open(config.dev_score_file, 'a+') as f_out:
        f_out.write(json.dumps(dict_eval))
        f_out.write('\n')

    return score_eval, dev_logits

def test_eval(config, model, batcher):
    '''
    Evaluates the accuracy on the test partition

    :param config:
    :param model:
    :param batcher:
    '''

    model.eval()
    dataset_reader = batcher.get_dataset_reader()
    test_writer = Writer(os.path.join(config.exp_dir, "test.json"), dataset_reader)

    with torch.no_grad():
        #added 
        pred_labels = []
        pred_logits = []
        t0 = time.time()
        for idx, batch in enumerate(batcher.get_test_batch()):
            t1 = time.time()
            pred_lbl, lbl_logits = model.predict(batch)

            #lbl_logits = lbl_logits.cpu().numpy()
              
            
            pred_labels.extend(pred_lbl.cpu().numpy().tolist())
            pred_logits.extend(lbl_logits.cpu().numpy().tolist())
          
            list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"][
                "idx"].cpu().numpy().tolist()
            list_lbl = batch["output"]["true_lbl"] if "true_lbl" in batch["output"] else batch["output"]["lbl"]

            if config.dataset.lower() == 'fewglue/record':
                list_idx = batch["input"]["qas_idx"]
                list_lbl = batch["input"]["candidate_entity"]
                test_writer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy())
            else:
                test_writer.add_batch(list_idx, pred_lbl, list_lbl, lbl_logits.cpu().numpy())
            
            #added
            t2 = time.time()
            diff1 = t1-t0
            diff2 = t2-t1
            diff3 = t2-t0
            #json_dict = {'start_loop':diff1, 'inside_loop':diff2, 'once_through':diff3}
            #writefile = 'time_difference/'+config.exp_dir
            #if not os.path.exists(writefile):
            #    os.makedirs(writefile)
            #print(writefile)
            #writefile = writefile+'time.json'
            #with open(writefile, "a") as f:
            #    f.write(json.dumps(json_dict)+ '\n')
    t3 = time.time()
    print('total inference time: {}'.format(t3-t0))
    #altered        
    #print(pred_logits)        
    test_writer.flush_file()
    return pred_labels, np.array(pred_logits)
    