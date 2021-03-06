import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.optimize import differential_evolution
from numpy.linalg import norm
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import scikitplot as skplt
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


def plotroc(y_true, y_pred, pos_label=2):
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label)
    # y_true =  # ground truth labels
    # y_probas =  # predicted probabilities generated by sklearn classifier
    skplt.metrics.plot_roc_curve(y_true, y_pred, curves=('each_class'))
    # skplt.metrics.plot_roc(y_true, y_pred, plot_micro=False, plot_macro=False)
    plt.show()


def plotpresionrecall(y_true, y_pred, logical=False):
    # skplt.metrics.plot_precision_recall_curve(y_true=y_true, y_probas=y_pred, plot_micro=logical)
    skplt.metrics.plot_precision_recall(y_true=y_true, y_probas=y_pred, plot_micro=logical)
    plt.show()


def conmatrix(cm, rscore=False, printresult=True):
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    score = (sensitivity + specificity) / 2
    if printresult:
        print(
            f'sensitivity= {sensitivity: .4f}\n specificity= {specificity: .4f} \n accuracy= {accuracy: .4f}\n score= {score: .4f}')
    if rscore:
        return score
    return (sensitivity, specificity, accuracy, score)


def voting(y_test, y_pred, NumberofCycle, title):
    n = np.size(NumberofCycle)
    N = np.cumsum(NumberofCycle)
    y_pred = y_pred.argmax(axis=-1)
    predict, test = list(), list()
    for k in range(n):
        if k == 0:
            out_true = y_test[0:N[k]]
            out_predict = y_pred[0:N[k] - 1]
            predict.append(np.mean(out_predict))
            test.append(np.mean(out_true))
        else:
            out_true = y_test[N[k - 1] + 1:N[k]]
            out_predict = y_pred[N[k - 1] + 1:N[k]]
            predict.append(np.mean(out_predict))
            test.append(np.mean(out_true))
    predict = [1 if x >= 0.7 else 0 for x in predict]
    cm = confusion_matrix(test, predict)
    print(f'-------------------{title}-------------------')
    print(cm)
    return conmatrix(cm)


def ensemble_predictions(weights, yhats):
    # make predictions

    # weighted sum across ensemble members
    summed = np.tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = np.argmax(summed, axis=-1)
    return result


def ensemble_predictions_prob(weights, yhats):
    # make predictions

    # weighted sum across ensemble members
    summed = np.tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    # result = np.argmax(summed, axis=1)
    return summed


# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(weights, yhats, testy):
    # make prediction
    yhat = ensemble_predictions(weights, yhats)
    yhat = yhat.astype(float)
    # calculate accuracy
    # cm = confusion_matrix(testy,yhat)
    # score = conmatrix(cm, rscore=True, printresult=False)
    return accuracy_score(testy, yhat)


# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# loss function for optimization process, designed to be minimized
def loss_function(weights, testX, testy):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(normalized, testX, testy)


def predict_model(model_name, x_test):
    loaded_model = keras.models.load_model(model_name)
    prob_result = loaded_model.predict(x_test)
    return prob_result


def predict_modelAll(models, x_test):
    out = list()
    for i, model in enumerate(models):
        out_temp = predict_model(model, x_test=x_test[1:, :])
        out.append(out_temp)
    return np.array(out)


def retun_modelname(iteration=1, alpha=1, foldername='result10times'):
    namebasedalpha = {1: 'first', 0.95: 'second', 0.90: 'third', 1.10: 'fourth'}
    iterationname = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
                     10: 'ten'}
    print(namebasedalpha[alpha])

    models = []
    xx = [0, 1, 2, 3, 4]
    for k in xx:
        temp_name = foldername + '/' + str(k) + '_' + str(namebasedalpha[alpha]) + '_model_' + str(
            iterationname[iteration]) + '.h5'
        models.append(temp_name)
    return models


def plot_ROC(y_test, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    n_classes = 3
    y_test1 = to_categorical(y_test)
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['red', 'blue', 'darkgreen'])
    plt.figure()
    lw = 3
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    # Plot all ROC curves

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right", fontsize='xx-large')
    plt.show()






def DE_algorithm(pred, true_value, n_members):
    # evaluate averaging ensemble (equal weights)
    weights = [1.0 / n_members for _ in range(n_members)]
    score = evaluate_ensemble(weights, pred, true_value)
    # print(f'accuarcy1 is {accu1}\n  accuarcy2 is {accu2}\n  accuarcy4 is {accu4}\n')
    print('Equal Weights Score: %.4f' % score)

    # define bounds on each weight
    bound_w = [(0.0, 1.0) for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (pred, true_value)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
    # get the chosen weights
    optimizse_weights = normalize(result['x'])
    print(f'Optimized Weights: {optimizse_weights}')
    # evaluate chosen weights
    # score = evaluate_ensemble(weights, pred, y_test[1:])
    # print('Optimized Weights Score: %.3f' % score)
    #wiehted_yhatequal = ensemble_predictions_prob(weights, pred)
    #wiehted_yhat = ensemble_predictions_prob(optimizse_weights, pred)
    return optimizse_weights, weights


def compuetvoting(pred1, y_test, n_members=3):
    wiehted_yhatequal_1, wiehted_yhat_1 = DE_algorithm(pred1, y_test, n_members=n_members)
    print('Equal weight....')
    print(classification_report(y_test, wiehted_yhatequal_1.argmax(axis=1)))
    print(confusion_matrix(y_test, wiehted_yhatequal_1.argmax(axis=1)))
    print('Optimize weight...')
    print(classification_report(y_test, wiehted_yhat_1.argmax(axis=1)))
    print(confusion_matrix(y_test, wiehted_yhat_1.argmax(axis=1)))
    plotpresionrecall(y_test, wiehted_yhatequal_1)
    plotpresionrecall(y_test, wiehted_yhat_1)

    plotroc(y_test, wiehted_yhatequal_1, pos_label=3)
    plotroc(y_test, wiehted_yhat_1, pos_label=3)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    n_classes = 3
    y_test1 = to_categorical(y_test)
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], wiehted_yhat_1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['red', 'blue', 'darkgreen'])
    plt.figure()
    lw = 3
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    # Plot all ROC curves

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right", fontsize='xx-large')
    plt.show()


