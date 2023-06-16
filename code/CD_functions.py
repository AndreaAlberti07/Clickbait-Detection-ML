import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd
from porter import stem
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skm



def remove_punctuation(text):
    '''Replace punctuation symbols with spaces'''
    
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text


def create_vocabulary(text_list, first_n, filename=None, store=False, remove_stopwords=False):
    '''Create a vocabulary from a list of texts. Each text is an array with string on each row'''
    
    #define the counter for the tokens
    cnt = Counter()
    
    #for each text in the list
    for text in text_list:
        
        # remove punctuation and split the text in tokens
        for row in text:
            row = remove_punctuation(row)
            tmp = row.strip().lower().split()
            cnt.update(tmp)
        
    if remove_stopwords is True:
        
        #load the stopwords
        stopwords = open('generated_gitignore/stopwords.txt', 'r').read().lower().split()
        
        #remove the stopwords and the words with length <= 2
        cnt = Counter({k: v for k, v in cnt.items() if k not in stopwords and len(k) > 2})
    
    #store in a filename the vocabulary
    if store is True:
        with open(filename, 'w') as f:
            for word, count in cnt.most_common(first_n):
                f.write(word + '\n')
        
    #return the n most common words
    return cnt.most_common(first_n)


def create_bow(text, _vocabulary, label):
    '''Create the bag of words for text (array of strings) using the vocabulary'''
    
    vectorizer = CountVectorizer(vocabulary=_vocabulary)
    bow_sparse = vectorizer.transform(text)
    bow_array = bow_sparse.toarray()

    # Add the label as the last column in the bow_array
    label_column = np.array([label] * len(bow_array)).reshape(-1, 1)
    bow_array = np.hstack((bow_array, label_column))

    return bow_array


def accuracy(predictions, labels):
    '''Compute the accuracy'''
    acc = np.sum(predictions == labels) / len(labels)
    return np.round(acc*100,2)


def plot_roc_curve(fpr, tpr, size):
    '''Plot the ROC curve'''
    plt.plot(fpr, tpr, label=f'voc_{size}')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.tight_layout()
    
    
def roc_curve_biases(model_funct, bow, w, b_range, size, title, plot=False):
    '''Compute and plot the ROC curve, changing the bias according to the values in the b_range array (n_biases x 2 (neg and pos))'''
    
    fprs = []
    tprs = []
    
    # Compute the predictions curve for each bias
    for b in b_range:
        if model_funct.__name__ == 'logreg_inference':
            probs = model_funct(bow[:,:-1], w, b)
            predictions_val = (probs > 0.5).astype(int)
            
        else:
            predictions_val, scores_val = model_funct(bow[:,:-1], w, b)

        # Compute Confusion Matrix
        confmat = skm.confusion_matrix(bow[:, -1], predictions_val)
        tn, fp, fn, tp = confmat.ravel()
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        fprs.append(fpr)
        tprs.append(tpr)
    
    if plot is True:
        
        # Plot the ROC curve
        plt.plot(fprs, tprs, label=f'voc_{size}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.legend()
        plt.tight_layout()
    
    return fprs, tprs


def cross_entropy(Prob,Y):
    '''Average cross entropy'''
    
    Prob=np.clip(Prob, 0.0001, 0.9999)
    ce = (-Y*np.log(Prob) - (1 - Y)*np.log(1-Prob)).mean()
    return ce


def logreg_inference(X, w, b):
    '''Inference for logistic regression'''
    
    logits = (X @ w) + b
    probability = np.exp(logits) / (1 + np.exp(logits))
    return probability


def logreg_training(X, Y, steps, lr, lambda_, tol=0, X_test=None, Y_test=None):
    '''Fit a logistic regression model using gradient descent. Return the weights, the bias, the train accuracies, the losses and the number of iterations until convergence'''
    
    m, n = X.shape

    w = np.zeros(n)
    b = 0
    
    train_accuracies = []
    test_accuracies = []
    losses = []
    iter_to_conv = 0
    
    for i in range(steps):
        Prob = logreg_inference(X, w, b)
        if X_test is not None:
            P_test = logreg_inference(X_test, w, b)
        
        if (i % 100) == 0:
            loss = cross_entropy(Prob,Y)
            prediction = (Prob > 0.5)
            accuracy = (prediction == Y).mean()
            train_accuracies.append(accuracy)
            losses.append(loss)
            if Y_test is not None:
                prediction_test = (P_test > 0.5)
                accuracy_test = (prediction_test == Y_test).mean()
                test_accuracies.append(accuracy_test)
        
        b = b - lr * (Prob-Y).sum()
        w = w - lr * ((Prob - Y) @ X + 2 * lambda_ * w)
        
        #if the loss is smaller than a threshold stop the training (early-stopping to reduce overfitting risk)
        if len(losses) > 1 :
            delta_loss = abs(losses[-1] - losses[-2])           
            if delta_loss < tol:
                break
        
        
        #compute the number of iteration until convergence
        iter_to_conv += 1
    return w, b, train_accuracies, test_accuracies, losses, iter_to_conv


def train_logreg_vocsizes(vocsizes, lr_, steps_, dest_dir, remove_stopwords=False):
    '''Train the logistic regression for different vocabulary sizes. Store the weights and the bias in a file'''

    for size in vocsizes:
        size = str(size)

        if remove_stopwords is True:
            # Load the BoW for training (vstack on the two classes)
            bow_train_bait = np.loadtxt('generated_gitignore/bow_train_bait_stop'+size+'.txt.gz')
            bow_train_nobait = np.loadtxt('generated_gitignore/bow_train_nobait_stop'+size+'.txt.gz')
            bow_train = np.vstack((bow_train_bait, bow_train_nobait))
            
            # Train the Logistic Regression
            w, b, _, _, _, _ = logreg_training(bow_train[:, :-1], bow_train[:, -1], steps = steps_, lr = lr_, lambda_=0)
        
            # Store the weights and the bias
            np.savez(dest_dir+'/param_logreg_stop'+size+'.npz', w=w, b=b)
        
        else:
            # Load the BoW for training (vstack on the two classes)
            bow_train_bait = np.loadtxt('generated_gitignore/bow_train_bait_NOstop'+size+'.txt.gz')
            bow_train_nobait = np.loadtxt('generated_gitignore/bow_train_nobait_NOstop'+size+'.txt.gz')
            bow_train = np.vstack((bow_train_bait, bow_train_nobait))

            # Train the Logistic Regression
            w, b, _, _, _, _ = logreg_training(bow_train[:, :-1], bow_train[:, -1], steps = steps_, lr = lr_, lambda_=0)
            
            # Store the weights and the bias
            np.savez(dest_dir+'/param_logreg_NOstop'+size+'.npz', w=w, b=b)
        
        print('size completed: ', size)
 
 
def fpr_accs_vocsizes(filename, title):
    '''Shows the results related to the lowest fpr for each vocsize, together with the accuracy for that vocsize'''
    
    #display the results related to the lowest fpr
    lowest_fpr=pd.read_csv(filename)

    # Create the figure 
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Vocabulary size')

    # Plot FPR on the first y-axis as a bar plot
    color = 'tab:red'
    ax1.set_ylabel('FPR (%)', color=color)
    ax1.bar(np.arange(len(lowest_fpr['vocsize'])) - 0.1, lowest_fpr['fpr']*100, color=color, label='FPR', width=0.2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create the second subplot (for accuracy)
    ax2 = ax1.twinx()

    # Plot accuracy on the second y-axis as a bar plot
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.bar(np.arange(len(lowest_fpr['vocsize'])) + 0.1, lowest_fpr['acc'], color=color, label='Accuracy', width=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Set the x-axis ticks and labels
    ax1.set_xticks(np.arange(len(lowest_fpr['vocsize'])))
    ax1.set_xticklabels(lowest_fpr['vocsize'])

    # Adjust the layout to prevent overlapping of bars and labels
    fig.tight_layout()

    # Add legends and title
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title(title)

    # Display the plot
    plt.tight_layout()
    plt.show()