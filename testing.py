import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter
import loggers
from model import Model
import statistics
import parsers

if __name__ == "__main__":

    args = parsers.parse_args()
    mean_scores_all = []
    std_scores_all = []
    N_retain_all = []
    logger = loggers.Loggers(args.filename, ['dataset size', 'run number', 'error prob', 'N retains'])

    # computing for each dataset sizes
    for size in args.sizes:

        with torch.no_grad():

            mean_scores = []
            N_retain = []

            # computing for x runs
            for i in range(args.run):

                # loading model
                model = Model(args)
                model.eval()
                # training
                model.training_loop(size, args)
                # testing
                mean_scores.append(model.testing_accuracy(args))
                N_retain.append(size * (1 - 2 * mean_scores[-1]))
                score = mean_scores[-1]
                logger.log([size, i, score, size * (1 - 2 * score)])

            # computing average error probability and std
            print('error probability is', statistics.mean(mean_scores))
            mean_scores_all.append(statistics.mean(mean_scores))
            std_scores_all.append(statistics.stdev(mean_scores))

            # number of images retain in memory
            N_retain_all.append(statistics.mean(N_retain))

    # plotting average error probability
    fig, ax = plt.subplots()
    ax.errorbar(args.sizes, mean_scores_all, std_scores_all)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("% 1.f"))
    plt.show()

    # plotting number of retained images vs number of showed images (log10 scale)
    fig, ax = plt.subplots()
    ax.plot(args.sizes, N_retain_all)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("% 1.1f"))
    plt.show()
