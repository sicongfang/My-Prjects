from utils import *
from dataset import load_data
from feature_extraction import get_features
from dependency import *
import numpy as np
import pickle
from train import *
from eval import *
from feature_extraction_da import *
from domain_adaptation import *

# The function
def test_model_da(file_name, domain):
    test_data = flatten_domain(load_data(file_name)[domain])
    with open('feature_dict_da.pkl', 'rb') as feature_f:
        feature_dict = pickle.load(feature_f)
    with open('model_da.pkl', 'rb') as model_f:
        model = pickle.load(model_f)

    all_predicted = []
    all_gold_arcs = []
    for id, sent_dict in test_data.items():
        gold_arcs = get_gold_arcs(sent_dict['HEAD'])
        projective = is_projective(gold_arcs, len(sent_dict['FORM']))
        if not projective or len(gold_arcs) == 1:
            continue
        all_gold_arcs.append(gold_arcs)

        predicted = predict_arcs(sent_dict, model, feature_dict)
        all_predicted.append(predicted)

    print_metrics(all_predicted, all_gold_arcs)

if __name__ == '__main__':
    training_data = flatten_domain(load_data('en-ud-train.conllu')['reviews'])
    print("Total number of sentences : "+str(len(training_data)))
    model = train_model_da(training_data)
    print("Saving model")
    with open('model_da.pkl','wb') as f:
        pickle.dump(model,f)
    print('\nEmail Results')
    test_model_da('en-ud-dev.conllu', 'email')
    print('\nNewsgroup Results:')
    test_model_da('en-ud-dev.conllu', 'newsgroup')

