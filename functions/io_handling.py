import argparse

# To make it possible for the user to input the following paramaters
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='dataset/data_train.csv', help='path/to/root')
    parser.add_argument('--test_dataset', type=str, default='dataset/data_test.csv', help='path/to/root')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size')
    parser.add_argument('--lr', type=int, default=0.00000000003, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=500, help='max train iteration')
    
    return parser.parse_args()

