import utils as ut
import argparse
import os


"""
Downloads 7 scenes training and testing data to the user supplied data_dir.
Data will be separated into two sets -- one for training/validation and one for testing.

If both testing/training are downloaded then this will download about 15gb of data, so it could 
take an hour or two to finish, depending on the connection
"""

train_dict = {
    'chess': [1, 2, 3,4,6],
    'fire': [1,2,3],
    'heads':[2],
    'office': [1, 2, 3, 4, 5, 6, 7, 8, 10],
    'pumpkin': [1, 2, 3, 6, 8],
    'redkitchen': [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13],
    'stairs': [1, 2, 3, 5, 6],
}

test_dict = {
    'chess': [5],
    'fire': [4],
    'heads':[1],
    'office':[9],
    'pumpkin':[7],
    'redkitchen':[14],
    'stairs': [4],
}

def download_7scenes(scene_dict, data_dir):
    for scene in scene_dict:
        seqs = scene_dict[scene]
        for seq in seqs:
            name = '{}_{}_mvs_training'.format(scene,seq)
            try:
                ut.download_and_unzip(name,data_dir)
            except Exception as e:
                print("Download of {} failed with exception {}".format(name,e))


def main(args):
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    if args.skip_train is not True:
        download_7scenes(train_dict,train_dir)
    if args.skip_test is not True:
        download_7scenes(test_dict, test_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory to download dataset to")
    parser.add_argument('--skip_train', action='store_true',
                        help="Will not download train data if flag is set")
    parser.add_argument('--skip_test', action='store_true',
                        help="Will not download test data if flag is set")
    args = parser.parse_args()
    main(args)



