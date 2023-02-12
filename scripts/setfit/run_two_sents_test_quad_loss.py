from torch.utils.data import DataLoader
import torch
from torch import nn, Tensor
import math
from typing import Iterable, Dict
import pandas as pd
import numpy as np
import torch.nn.functional as F
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from datasets import load_dataset, Dataset
from sentence_transformers.losses import SentenceTransformer, InputExample, CosineSimilarityLoss
from sentence_transformers import SentencesDataset


from sentence_transformers import SentenceTransformer


class Mode(Enum):
    FOUR_SENTS = 1
    CONCAT = 2
   

class DistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class QuadLoss_New(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.
    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).
    Example::
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity(),  distance_metric=DistanceMetric.EUCLIDEAN):
        super(QuadLoss_New, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.distance_metric = distance_metric
        self.margin = 0.0

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
       
        distance_1 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        distance_2 = self.cos_score_transformation(torch.cosine_similarity(embeddings[2], embeddings[3]))

        loss_1 = 1-((1.0-labels[:,0])*distance_1+labels[:,0]*(1.0-distance_1))
        loss_2 = 1-((1.0-labels[:,1])*distance_2+labels[:,1]*(1.0-distance_2))
        losses = 0.5*loss_1+0.5*loss_2+self.margin

        dist_3 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[2]))
        dist_4 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[3]))
        dist_5 = self.cos_score_transformation(torch.cosine_similarity(embeddings[1], embeddings[2]))
        dist_6 = self.cos_score_transformation(torch.cosine_similarity(embeddings[1], embeddings[3]))
        
     
        #output = output*labels.view(-1)+(1-output)*(1-labels.view(-1))+0.0
        #output = distance_1-distance_2
        #output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0],embeddings[1])-torch.cosine_similarity(embeddings[2],embeddings[3]))
        #output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0]-embeddings[1],embeddings[2]-embeddings[3]))
        #losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        #out = self.loss_fct(losses, labels.view(-1))
        #return losses.mean()

        out = torch.mean(losses+ 2.0*(dist_3+dist_4+dist_5+dist_6))
        return out    

class QuadLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.
    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).
    Example::
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity(),  distance_metric=DistanceMetric.EUCLIDEAN):
        super(QuadLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.distance_metric = distance_metric
        self.margin = 0.5

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
       
        distance_1 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        distance_2 = self.cos_score_transformation(torch.cosine_similarity(embeddings[2], embeddings[3]))

        dist_3 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[2]))
        dist_4 = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[3]))
        dist_5 = self.cos_score_transformation(torch.cosine_similarity(embeddings[1], embeddings[2]))
        dist_6 = self.cos_score_transformation(torch.cosine_similarity(embeddings[1], embeddings[3]))
        
        distances = torch.abs(distance_1-distance_2)
        #output = output*labels.view(-1)+(1-output)*(1-labels.view(-1))+0.0
        #output = distance_1-distance_2
        #output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0],embeddings[1])-torch.cosine_similarity(embeddings[2],embeddings[3]))
        #output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0]-embeddings[1],embeddings[2]-embeddings[3]))
        #losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        #out = self.loss_fct(losses, labels.view(-1))
        #return losses.mean()
        out = self.loss_fct(distances, labels.view(-1))+torch.mean(dist_3+dist_4+dist_5+dist_6)
        
        return out
        

def dual_sentence_pairs_generation(sentences1, sentences2, labels, dual_pairs, mode):
    
    num_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in num_classes]

    for first_idx in range(len(sentences1)):
        curr_sentence1 = sentences1[first_idx]
        curr_sentence2 = sentences2[first_idx]
        label = labels[first_idx]
        #debug do not take 2 negative samples 
        
        second_idx = np.random.choice(idx[np.where(num_classes == label)[0][0]])
        positive_sentence1 = sentences1[second_idx]
        positive_sentence2 = sentences2[second_idx]
        # Prepare 2 positive pairs and update the sentences and labels
        # lists, respectively

        if mode==Mode.FOUR_SENTS:
            #dual_pairs.append(InputExample(texts=[curr_sentence1, curr_sentence2, positive_sentence1, positive_sentence2], label=0.0))
            dual_pairs.append(InputExample(texts=[curr_sentence1, curr_sentence2, positive_sentence1, positive_sentence2], label=[float(label),float(label)]))
        if mode==Mode.CONCAT:
            dual_pairs.append(InputExample(texts=[curr_sentence1+". "+curr_sentence2, positive_sentence1+". "+positive_sentence2], label=1.0))

        negative_idx = np.where(labels != label)[0]
        third_idx= np.random.choice(negative_idx)
        negative_sentence1 = sentences1[third_idx]
        negative_sentence2 = sentences2[third_idx]
        # Prepare a negative pair of sentences and update our lists
        if mode==Mode.FOUR_SENTS:
            #dual_pairs.append(InputExample(texts=[curr_sentence1, curr_sentence2, negative_sentence1, negative_sentence2], label=1.0)) 
            dual_pairs.append(InputExample(texts=[curr_sentence1, curr_sentence2, negative_sentence1, negative_sentence2], label=[float(label),float(labels[third_idx])])) 
        if mode==Mode.CONCAT:
            dual_pairs.append(InputExample(texts=[curr_sentence1+". "+curr_sentence2, negative_sentence1+". "+negative_sentence2], label=0.0)) 

    return dual_pairs


def setfit_two_sents_quad_loss(args, train_data, eval_data, metric):
   
    x_train_sent1 = train_data["sentence1"]
    x_train_sent2 = train_data["sentence2"]
    y_train = train_data["label"]
   
    x_eval_sent1 = eval_data["sentence1"]
    x_eval_sent2 = eval_data["sentence2"]
    y_eval = eval_data["label"]

    train_examples = []
    for _ in range(args.num_iterations):
         train_examples = dual_sentence_pairs_generation(np.array(x_train_sent1), np.array(x_train_sent2), np.array(y_train), train_examples, Mode.FOUR_SENTS)
    
    model = SentenceTransformer(args.model)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    #train_loss = QuadLoss(model=model)
    train_loss = QuadLoss_New(model=model)
    #train_loss = CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up

    # traine sentence transformer
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          #evaluator=dev_evaluator,
          epochs=args.num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.1),
          warmup_steps=warmup_steps,
          #output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )

    # traine model head

    # cos_sim =[]
    # for text1,text2,label in zip(x_eval_sent1,x_eval_sent2,y_eval):
    #     emb1=model.encode(text1)
    #     emb2=model.encode(text2)
    #     cos_sim.append([float(util.cos_sim(emb1, emb2)),0])

    embeddings1 = model.encode(x_train_sent1)
    embeddings2 = model.encode(x_train_sent2)
    cos_sim=(torch.cosine_similarity(Tensor(embeddings1),Tensor(embeddings2))).reshape(-1, 1)
    #cos_sim = util.cos_sim(embeddings1, embeddings2)
    sgd = LogisticRegression()
    sgd.fit(cos_sim, y_train)

    embeddings1 = model.encode(x_eval_sent1)
    embeddings2 = model.encode(x_eval_sent2)
    cos_sim=(torch.cosine_similarity(Tensor(embeddings1),Tensor(embeddings2))).reshape(-1, 1)
    y_pred_eval_sgd = sgd.predict(cos_sim)

    setfit_result = metric.compute(predictions=y_pred_eval_sgd, references=y_eval)

    #print('Acc = ', round(accuracy_score(y_eval, y_pred_eval_sgd),4))

    # No fit
    model_no_fit = SentenceTransformer('paraphrase-mpnet-base-v2')

    embeddings1 = model_no_fit.encode(x_train_sent1)
    embeddings2 = model_no_fit.encode(x_train_sent2)
    cos_sim=(torch.cosine_similarity(Tensor(embeddings1),Tensor(embeddings2))).reshape(-1, 1)
    sgd = LogisticRegression()
    sgd.fit(cos_sim, y_train)

    embeddings1 = model_no_fit.encode(x_eval_sent1)
    embeddings2 = model_no_fit.encode(x_eval_sent2)
    cos_sim=(torch.cosine_similarity(Tensor(embeddings1),Tensor(embeddings2))).reshape(-1, 1)

    y_pred_eval_sgd = sgd.predict(cos_sim)
    no_fit_result = metric.compute(predictions=y_pred_eval_sgd, references=y_eval)
    #print('Acc Not Fit = ', round(accuracy_score(y_eval, y_pred_eval_sgd),4))
   
    return round(setfit_result["accuracy"],3), round(no_fit_result["accuracy"],3)
    

   

