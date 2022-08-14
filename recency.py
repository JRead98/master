import torch
from matplotlib import pyplot as plt
import loggers
import parsers
from model import Model

if __name__ == "__main__":

    args = parsers.parse_args()
    logger = loggers.Loggers(args.filename, ['dataset size', 'run number', 'image number', 'score'])

    for size in args.sizes:

        with torch.no_grad():

            scores = None

            # computing recency on X runs
            for i in range(args.run):

                # training
                model = Model(args)
                model.eval()
                # chose dataset size
                model.training_loop(args.sizes[0], args)
                # testing for recency
                tmp = model.recency(args)

                for j, k in enumerate(tmp):
                    logger.log([size, i, j, k])

                if scores is None:
                    scores = tmp
                else:
                    scores += tmp
            scores /= args.run

            # plotting recency
            plt.plot(scores)
            plt.show()
