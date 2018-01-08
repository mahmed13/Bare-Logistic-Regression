import numpy as np
import pandas as pd
import argparse
import math
import pickle

STEP_SIZE = 0.0001

if __name__ == "__main__":
    # set up command line args
    parser = argparse.ArgumentParser(description = "Logistic Regression",
        add_help = "How to use",
        prog = "python LR.py [train-data] [train-label] [test-data]")
    parser.add_argument("paths", nargs = 4)
    args = vars(parser.parse_args())

    # Access command-line arguments as such:
    # read in data
    df_train = pd.read_csv(args["paths"][0], delimiter=' ', header=None)
    df_train_label = pd.read_csv(args["paths"][1], delimiter=' ', header=None)
    df_test = pd.read_csv(args["paths"][2], delimiter=' ', header=None)
    df_test_label = pd.read_csv(args["paths"][3], delimiter=' ', header=None)

    # transform data to matrix
    X_train = df_train.as_matrix()
    Y_train = df_train_label.as_matrix()
    X_test = df_test.as_matrix()
    Y_test = df_test_label.as_matrix()


    # compute result of sigmoid function from numpy array
    def sigmoid_function(x):
        y = 1. / (1. + (np.exp(-x)))
        return y

    # compute objective function from numpy array
    def objective_function(x):
        return np.sum(np.log(1 + np.exp(-x)))

    def convertSparse(sparse):
        n_documents = np.max(sparse[:, 0])


        #[doc, word, occurances]
        starting_doc_n = np.min(sparse[:, 0])
        starting_word_n = np.min(sparse[:, 1])
        mat = np.zeros([n_documents-starting_doc_n+1, 10770])

        for i in sparse:
            mat[i[0]-starting_doc_n][i[1]-starting_word_n] = i[2]
        return mat

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    ## change
    def log_likelihood(X, Y, b):
        Xb = np.dot(X, b)
        print('ll=',Xb)
        log_l = np.sum(Y * Xb - np.log(1 + np.exp(Xb)))
        return log_l

    ## change
    def logistic_regression(X, Y, num_steps, step_size):

        # append dummy variable to X

        b_0 = np.ones((X.shape[0], 1))
        X = np.hstack((b_0, X))

        # initialize weights
        weights = np.ones(X.shape[1]).reshape(-1, 1)

        # step through gradient descent
        for step in range(num_steps):
            scores = np.dot(X, weights)
            predictions = sigmoid(scores).reshape(-1,1)
            # calculate error and gradient
            output_error_signal = Y - predictions
            gradient = np.dot(X.T, output_error_signal)


            print('Y', Y.shape)
            print('output', output_error_signal.shape)

            # update weights
            weights += step_size * gradient

            # print out objective value
            if step % math.floor(num_steps/5) == 0:
                pass
                #print(log_likelihood(X, Y, weights))

        return weights


    # saving pickles
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    # loading pickles
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # convert dense sparse matrix to normal matrix
    X_train = convertSparse(X_train)
    X_test = convertSparse(X_test)


    # account for dummy variable X*BetaT with intercept
    X_test = np.hstack((np.ones([X_test.shape[0],1]), X_test))


    weights = logistic_regression(X_train, Y_train, num_steps=30000, step_size=STEP_SIZE)
    #save_obj(weights,'weights')

    #weights = load_obj('weights')

    final_scores = np.dot(X_test, weights)
    pred= np.round(sigmoid(final_scores))

    for i in range(len(pred)):
        print(int(pred[i][0]))
    #print('Accuracy: {0}'.format((pred == Y_test).sum().astype(float) / len(preds)))




