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
    logger = loggers.Loggers(args.filename, ['run number', 'error prob'])

    with torch.no_grad():

        mean_scores = []

        # computing for x runs
        for i in range(args.run):

            # loading model
            model = Model(args)
            model.load_state_dict(torch.load('weights_only.pth'))
            model.eval()
            # testing
            mean_scores.append(model.threshold(args))
            score = mean_scores[-1]
            logger.log([i, score])

        # computing average error probability and std
        print('error probability is', statistics.mean(mean_scores))
        mean_scores_all.append(statistics.mean(mean_scores))
        std_scores_all.append(statistics.stdev(mean_scores))


    # plotting average error probability
    fig, ax = plt.subplots()
    ax.errorbar(args.sizes, mean_scores_all, std_scores_all)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("% 1.f"))
    plt.show()
