import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis


def plot_hist(ser, form_x=False, form_y=False, width=14, height=5,
              min_x=None, max_x=None, min_y=None, max_y=None,
              kde=False, rug=False, x_label=None, bins=None, logx=False,
              skew_kurt=True, plot_mean=True, plot_median=True, sdev=True,
              mean_xlift=1.1, med_xlift=0.7, sdev_xlift=1.3, skew_xlift=2, kurt_xlift=1.3, skew_kurt_rot=30,
              title='Distribution', title_size=20,
              x_tick_size=16, y_tick_size=16, x_lab_size=20, y_lab_size=20, mean_med_size=18,
              act='show', save_path='distribution.png', dpi=300, save_only=True):
    """
    plot distribution of the provided series
    :param ser: numpy array or pandas Series
    series from which to plot distributions
    :param form_x: boolean
    whether to add thousands separator to the x tick labels
    :param form_y: boolean
    whether to add thousands separator to the y tick labels
    :param kde: boolean
    whether to plot kernel density estimation (default = histogram)
    :param x_label: string
    label to use for x axis
    :param plot_mean: boolean
    whether to plot the mean of the series
    :param plot_median: boolean
    whether to plot the median of the series
    :param mean_xlift: float
    caption lift along the x axis for the mean
    :param med_xlift: float
    caption lift along the x axis for the median
    :param sdev: boolean
    whether to plot the standard deviation of the series
    :param sdev_xlift: float
    caption lift along the x axis for standard deviation
    :param title: string
    plot title
    :param title_size: float
    fontsize to use for the plot title
    :param x_tick_size: float
    fontsize to use for x ticks
    :param y_tick_size: float
    fontsize to use for y ticks
    :param x_lab_size: float
    fontsize to use for x axis label
    :param y_lab_size: float
    fontsize to use for y axis label
    :param mean_med_size: float
    fontsize to use for mean, median and standard deviation
    :param act: string ('show' or 'save')
    whether to show or save the plot
    :param save_path: string
    where to save the plot (relative from script location)
    :param dpi: int
    resolution for saving the plot
    :param save_only: boolean
    save without displaying
    :return: None, plots and displays or saves the result
    """
    # create figure and axis
    f, ax = plt.subplots(1, figsize=(width, height))

    if logx:
        hist, bins, _ = plt.hist(ser, bins=bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(ser, bins=logbins)
        ax.set_xscale('log')
    else:
        # plot distribution
        sns.distplot(ser, kde=kde, rug=rug, bins=bins, ax=ax)
        if form_x:
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # plot mean of the series
    if plot_mean:
        mean = ser.mean()
        ax.axvline(mean, linestyle='--', color='deeppink')
        ax.text(mean * mean_xlift, 0, 'Mean: {0:,.2f}'.format(mean), fontsize=mean_med_size, rotation=90)
    # plot median of the series
    if plot_median:
        median = ser.median()
        ax.axvline(median, linestyle='--', color='teal')
        ax.text(median * med_xlift, 0, 'Median: {0:,.2f}'.format(median), fontsize=mean_med_size, rotation=90)
    # print standard deviation of the series
    if sdev:
        mean = ser.mean()
        ax.text(mean * sdev_xlift, 0, 'StDev: {0:,.2f}'.format(ser.std()), fontsize=mean_med_size, rotation=90)
    # print excess kurtosis and skewness
    if skew_kurt:
        ax.text(mean * skew_xlift * sdev_xlift, 0, 'skewness: {0:.2f}'.format(skew(ser)),
                fontsize=mean_med_size, rotation=skew_kurt_rot)
        ax.text(mean * skew_xlift * kurt_xlift * sdev_xlift, 0, 'excess kurtosis : {0:.2f}'.format(kurtosis(ser)),
                fontsize=mean_med_size, rotation=skew_kurt_rot)

    # format axes
    if form_y:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # configure axes parameters
    ax.set_title(title, fontsize=title_size)
    plt.xticks(fontsize=x_tick_size)
    plt.yticks(fontsize=y_tick_size)
    plt.grid(axis='x')
    if kde:
        ax.set_ylabel('Kernel density estimation (KDE)', fontsize=y_lab_size)
    else:
        ax.set_ylabel('Count of records', fontsize=y_lab_size)

    if x_label:
        ax.set_xlabel(x_label, fontsize=x_lab_size)

    if min_x or max_x:
        ax.set_xlim(left=min_x, right=max_x)
    if min_y or max_y:
        ax.set_ylim(bottom=min_y, top=max_y)

    # save or show results
    if act == 'show':
        plt.show()
    elif act == 'save':
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print("Saved output plot to", save_path)
        if save_only:
            plt.close(f)
