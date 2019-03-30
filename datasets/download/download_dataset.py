import boto3
import botocore
import constants
import utils as ut
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help = "Name of dataset to be downloaded")
    parser.add_argument('data_dir', type=str, help = "Diretory to download dataset to")
    args = parser.parse_args()
    ut.download_and_unzip(args.name, args.data_dir)
