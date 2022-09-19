import os
import json
import argparse


GLUE_DATASETS = ['SetFit/stsb', 'SetFit/mnli_mm', 'SetFit/mnli', 'SetFit/wnli', 
                 'SetFit/qnli', 'SetFit/mrpc', 'SetFit/rte', 'SetFit/qqp']
                 
AMZ_MULTI_LING = ['SetFit/amazon_reviews_multi_ja','SetFit/amazon_reviews_multi_zh', 'SetFit/amazon_reviews_multi_de', 
              'SetFit/amazon_reviews_multi_fr', 'SetFit/amazon_reviews_multi_es', 'SetFit/amazon_reviews_multi_en']

INTENT_MULTI_LING = ['SetFit/amazon_massive_intent_ar-SA','SetFit/amazon_massive_intent_es-ES', 'SetFit/amazon_massive_intent_de-DE', 
                    'SetFit/amazon_massive_intent_ja-JP', 'SetFit/amazon_massive_intent_zh-CN', 'SetFit/amazon_massive_intent_ru-RU']

SINGLE_SENT_DATASETS = ['SetFit/sst2', 'SetFit/sst5', 'SetFit/imdb', 'SetFit/subj', 'SetFit/ag_news', 'SetFit/bbc-news',
                    'SetFit/enron_spam', 'SetFit/student-question-categories', 'SetFit/TREC-QC', 'SetFit/toxic_conversations',
                    'SetFit/amazon_counterfactual_en', 'SetFit/CR', 'SetFit/SentEval-CR', 'SetFit/emotion', 'SetFit/amazon_polarity', 'SetFit/ade_corpus_v2_classification']
                
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def write_seed_output(pretrained_weight, task_name, sample_size, ds_seed, metric, english, prompt):
    dataset = task_name[7:] #remove setfit/
    if dataset in ["toxic_conversations"]:
        json_dict = {"measure": "ap", "score": metric}
    
    elif dataset in ["amazon_counterfactual_en"]:
        json_dict = {"measure": "matthews_correlation", "score": metric}
    
    elif 'SetFit/' + dataset in AMZ_MULTI_LING:
        json_dict = {"measure": "mean_absolute_error", "score": metric}        
    
    else:
        json_dict = {"measure": "acc", "score": metric}
    
    if 'microsoft/' in pretrained_weight:
        pretrained_weight = pretrained_weight.replace('microsoft/', '')
    
    if task_name in SINGLE_SENT_DATASETS:
        writefile = 'seed_output/' + pretrained_weight +'/'+ dataset +'/'+ 'train-'+str(sample_size)+'-'+str(ds_seed)+ '/'
    else:
        if english:
            lang = 'eng'
        else:
            lang = 'lang'
        if prompt:
            prompting = 'prompt'
        else:
            prompting = 'no-prompt'
        writefile = 'seed_output/' + pretrained_weight + "__"+lang+'_'+prompting +'/'+ dataset +'/'+ 'train-'+str(sample_size)+'-'+str(ds_seed)+ '/'
    if not os.path.exists(writefile):
        os.makedirs(writefile)
    writefile = writefile+'results.json'
    with open(writefile, "a") as f:
        f.write(json.dumps(json_dict))

def fix_train_amzn(dataset, lang_star_dict):
    dataset = dataset.rename_column("label_text", "str_label_text")
    label_text =[lang_star_dict[i] for i in dataset['label']]
    dataset = dataset.add_column('label_text', label_text)
    return dataset

def fix_amzn(dataset, lang_star_dict):
    dataset = dataset.rename_column("label_text", "str_label_text")
    for split, dset in dataset.items():
        label_text =[lang_star_dict[i] for i in dset['label']]
        dset = dset.add_column('label_text', label_text)
        dataset[split] = dset
    return dataset

def fix_intent(task_name, dataset, english):
    dataset = dataset.rename_column("label_text", "str_label_text")
    if english:
        for split, dset in dataset.items():
            label_text = []
            for txt_lab in dset["str_label_text"]:
                label_text.append(txt_lab.replace("_", " "))
            dset = dset.add_column('label_text', label_text)
            dataset[split] = dset
        
        lang_pattern = '[TEXT1] this is [LBL]'        
    else:
        if task_name == 'SetFit/amazon_massive_intent_zh-CN':
            lang_pattern = '[TEXT1] 这是 [LBL]'
            dataset = dataset.rename_column("label_text_ch", "label_text")
        
        elif task_name == 'SetFit/amazon_massive_intent_ru-RU':
            lang_pattern = '[TEXT1] это [LBL]'
            dataset = dataset.rename_column("label_text_ru", "label_text")
        
        elif task_name == 'SetFit/amazon_massive_intent_de-DE':
            lang_pattern = '[TEXT1] dies ist [LBL]'
            dataset = dataset.rename_column("label_text_de", "label_text")        
        
        elif task_name == 'SetFit/amazon_massive_intent_ja-JP':
            lang_pattern = '[TEXT1]これは[LBL]だ'
            dataset = dataset.rename_column("label_text_jp", "label_text")
        
        elif task_name == 'SetFit/amazon_massive_intent_es-ES':
            lang_pattern = '[TEXT1] esto es [LBL]'
            dataset = dataset.rename_column("label_text_es", "label_text")
        
    return dataset, lang_pattern

def multiling_verb_pattern(task_name, english, prompt):
    assert task_name in AMZ_MULTI_LING
    if not english:
        if task_name == 'SetFit/amazon_reviews_multi_zh':
            lang_star_dict = {0: '1星', 1: '2星', 2: '3星', 3: '4星', 4: '5星'}
            lang_pattern = '[TEXT1] 这是 [LBL]'
        
        elif task_name == 'SetFit/amazon_reviews_multi_de':
            lang_star_dict = {0: '1 stern', 1: '2 sterne', 2: '3 sterne', 3: '4 sterne', 4: '5 sterne'}
            lang_pattern = '[TEXT1] dies ist [LBL]'
        
        elif task_name == 'SetFit/amazon_reviews_multi_fr':
            lang_star_dict = {0: '1 étoile', 1: '2 étoiles', 2: '3 étoiles', 3: '4 étoiles', 4: '5 étoiles'}
            lang_pattern = '[TEXT1] est noté [LBL]'
        
        elif task_name == 'SetFit/amazon_reviews_multi_ja':
            lang_star_dict = {0: '一つ星', 1: '二つ星', 2: '三つ星', 3: '四つ星', 4: '五つ星'}
            lang_pattern = '[TEXT1]これは[LBL]だ'
        
        elif task_name == 'SetFit/amazon_reviews_multi_es':
            lang_star_dict = {0: '1 estrella', 1: '2 estrellas', 2: '3 estrellas', 3: '4 estrellas', 4: '5 estrellas'}
            lang_pattern = '[TEXT1] esto es [LBL]'
    else:
        lang_star_dict = {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
        lang_pattern = '[TEXT1] this is [LBL]'
    
    if prompt:
        return lang_star_dict, lang_pattern
    
    else:
        lang_pattern = '[TEXT1] [LBL]'
        return lang_star_dict, lang_pattern

def fix_stsb(dataset):
    dataset = dataset.rename_column("label", "float_label")
    dataset = dataset.rename_column("label_text", "na_label_text")
    sim_dict = {0: 'very different', 1: 'different', 2: 'dissimilar', 3: 'somewhat similar', 4: 'similar', 5: 'very similar'}
    for split, dset in dataset.items():
        if split == 'test':
            continue
        else:
            label = [round(i) for i in dset['float_label']]
            dset = dset.add_column("label", label)
            label_text = [sim_dict[i] for i in label]
            dset = dset.add_column('label_text', label_text)
            dataset[split] = dset
    return dataset


def write_evaluation_json(accs, mics, macs, avg_pres, logit_aps, num_labs, sample_size, task_name, configs, english, prompt):
    if sample_size in ["full", 500]:
        assert len(accs) == len(mics) == len(macs) == len(avg_pres) == len(logit_aps) == len(ADAPET_SEEDS)
    else:
        assert len(accs) == len(mics) == len(macs) == len(avg_pres) == len(logit_aps) == len(SEEDS)
    
    round_to = 10
    mean_acc = round(np.mean(accs), round_to)
    acc_std = round(np.std(accs), round_to)

    mean_micro = round(np.mean(mics), round_to)
    micro_std = round(np.std(mics), round_to)

    mean_macro = round(np.mean(macs), round_to)
    macro_std = round(np.std(macs), round_to)
    
    mean_avg_pre = round(np.mean(avg_pres), round_to)
    avg_pre_std = round(np.std(avg_pres), round_to)
    
    mean_logit_ap = round(np.mean(logit_aps), round_to)
    logit_ap_std = round(np.std(logit_aps), round_to)
    
    #in the multiclass scenario, average precision is not defined
    if num_labs > 2:
        mean_avg_pre = 'NA'
        avg_pre_std = 'NA' 
        mean_logit_ap = 'NA'
        logit_ap_std = 'NA'

    json_dict = {
        "mean_acc": mean_acc,
        "acc_std": acc_std,
        "mean_f1_mic": mean_micro,
        "f1_mic_std": micro_std,
        "mean_f1_mac": mean_macro,
        "f1_mac_std": macro_std,
        "mean_avg_pre": mean_avg_pre,
        "avg_pre_std": avg_pre_std,
        "mean_logit_ap": mean_logit_ap,
        "logit_ap_std": logit_ap_std,
    }
    write_dir = 'results/'+ configs["pretrained_weight"] + '/' + task_name.lower()[7:]
    
    if english:
        write_dir = write_dir + '_eng'
    else:
        write_dir = write_dir + '_lang'
    if prompt:
        write_dir = write_dir + '_prompt'
    else:
        write_dir = write_dir + '_no_prompt'

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    writefile = write_dir + "/" + str(sample_size) + "_split_results.json"
    with open(writefile, "w") as f:
        f.write(json.dumps(json_dict) + "\n")

