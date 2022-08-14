import torch
import parsers
from model import Model

if __name__ == "__main__":

    args = parsers.parse_args()

    with torch.no_grad():

        # loading model
        model = Model(args)
        model.eval()
        # training
        model.training_loop(args.sizes[0], args)
        model.histogram(args)

        # saving weights
        torch.save(model.state_dict(), 'weights_only.pth')
