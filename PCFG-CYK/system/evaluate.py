import numpy as np
from PYEVALB import parser as evalbparser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt




def score(reference_parse, proposed_parse):
    """ Performs evaluation on a single parse tree

    Args:
        reference_parse (str): reference parse tree for the current sentence
        proposed_parse (str): proposed parse tree for the current sentence
    Returns:
        precision, recall, f_score, accuracy
        sh1: length of the predicted sentence
        sh2: length of the true sentence

    """
    true_tree = evalbparser.create_from_bracket_string(reference_parse)
    test_tree = evalbparser.create_from_bracket_string(proposed_parse)

    y_true = np.array(true_tree.poss)
    y_pred = np.array(test_tree.poss)

    sh1 = y_pred.shape[0]
    sh2 = y_true.shape[0]

    y_pred = (y_true == y_pred) * 1
    y_true = np.ones(len(y_true))
    
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1])
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f_score, accuracy, sh1, sh2


def evaluation(groundtruth_filename='evaluation_data.ground_truth', prediction_filename='evaluation_data.parser_output'):
    """ Evaluates the proposed parses and compares them with the reference ground_truth file

    Computes mean accuracy, precision, recall, F1 score and plots the errors done

    Args:
        groundtruth_filename (str): path for reference file name
        prediction_filename (str): path for prediction file name
        
    """
    truth_corpus = []
    with open(groundtruth_filename) as f:
        for i, l in enumerate(f):
            truth_corpus.append(l)

    test_corpus = []
    with open(prediction_filename) as f:
        for i, l in enumerate(f):
            test_corpus.append(l)

    precisions, recalls, f_scores, accuracies, errors, len_sent = [], [], [], [], [], []
    for truth, test in zip(truth_corpus, test_corpus):
        precision, recall, f_score, accuracy, sh1, sh2 = score(truth, test)
        len_sent.append(sh2)
        if sh1==1 and sh2!=1:
            errors.append(sh2)
        
        precisions.append(float(precision))
        recalls.append(float(recall))
        f_scores.append(float(f_score[0]))
        accuracies.append(accuracy)

    print('Precision : {:.2f}%'.format(np.mean(precisions) * 100))
    print('Recall : {:.2f}%'.format(np.mean(recalls) * 100))
    print('F1 score : {:.2f}%'.format(np.mean(f_scores) * 100))
    print('Accuracy : {:.2f}%'.format(np.mean(accuracies) * 100))

    plt.figure()
    plt.hist(len_sent, bins = max(len_sent), alpha=0.5, histtype='bar', ec='black', label='distribution of lengths in the test set')
    plt.hist(errors, bins = max(errors), alpha=0.8, histtype='bar', ec='black', label='errors of the system')
    plt.legend()
    plt.xlabel('length of the sentence')
    plt.title('number of errors and number of sentences per length')
    plt.show()
