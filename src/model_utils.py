import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from time import time


def plot_validation_curve(classifier, model_name, X, y, param_name, param_range, n_jobs=1, cv=10,
                          fig_height=4, fig_width=6, xlog=False):
    t = time()

    train_scores, test_scores = validation_curve(estimator=classifier, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                                 param_name=param_name, param_range=param_range)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(1, figsize=(fig_width, fig_height))

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.legend(loc='lower right')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    ax.set_title("{0}, validation curve\n{1}: {2}".format(model_name, param_name,
                                                          list(pd.Series(param_range).apply(lambda x: round(x, 3)))))
    plt.legend(loc='lower right')
    if xlog:
        plt.xscale('log')

    elapsed = time() - t
    print("Validation curve for {0} plotted, took {1:,.2f} seconds ({2:,.2f} minutes)"
          .format(model_name, elapsed, elapsed / 60))
    plt.show()