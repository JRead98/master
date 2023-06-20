from argparse import ArgumentParser

def parse_args():

    # parsers
    parser = ArgumentParser()
    
    # dataset sizes
    parser.add_argument('--sizes', nargs='+', default=[20, 40, 100, 200, 400, 1000, 4000, 10000], type=int)
    
    # dataset path
    parser.add_argument('--datapath', default='C:/Users/johnr/Desktop/Testing/256_ObjectCategories')
    
    # choosing only jpg images
    parser.add_argument('--only_jpg', action='store_true')
    
    # learning rate
    parser.add_argument('--lr', default=0.01, type=float)
    
    # generate random weights between x and y
    parser.add_argument('--min_weights', default=-1, type=float)
    parser.add_argument('--max_weights', default=1, type=float)
    
    # runs number
    parser.add_argument('--run', default=100, type=int)
    
    # learning rule
    parser.add_argument('--learning_rule', default='Hebbian')
    
    # CNN
    parser.add_argument('--model', default='resnet')
    
    # filename
    parser.add_argument('--filename', default='simulation.csv')

    return parser.parse_args()
