import boto3
import botocore
import constants as c
import tarfile
import os
s3 = boto3.resource('s3')

def download_data(name, data_dir):
    file_name = '{}.tar.gz'.format(name)
    key = os.path.join(c.DATA_PREFIX,file_name)
    path = os.path.join(data_dir, file_name)
    print("Downloading dataset {} to dir {}".format(name, data_dir))
    try:
        s3.Bucket(c.U6_DATASET_BUCKET).download_file(key, path)
        return path
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
            return None
        else:
            raise


def unzip_file(fpath, extraction_path = '.', strip_prefix = True, cleanup=True):
    print("Extracting file {}".format(fpath))
    tar = tarfile.open(fpath)
    prefix = '/tmp' # this prefix is added from the u6 Dataset upload process
    for member in tar.getmembers():
        if strip_prefix:
            member.name = member.name.lstrip(prefix)
        tar.extract(member, extraction_path)
    tar.close()
    if cleanup:
        os.remove(fpath)


def download_and_unzip(name, data_dir):
    fpath = download_data(name, data_dir)
    unzip_file(fpath, data_dir)

    







