import json
from urllib import response
import pandas as pd
import os
import spacy

import requests
import tarfile
from tqdm import tqdm

"""
Method to get the size (bytes) of a directory (including all of its child directory)
Code taken from:
https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
"""
def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    return total_size # bytes

"""
Method to convert bytes in int into a a human recognizable size in string
Dynamically change the unit of the size, depending on how big it is
"""
def get_human_readable_size(size, precision=2):
    suffixes= ['B','KB','MB','GB','TB'] # list of unit
    suffixIndex = 0 # currently choosing from B
    while size > 1024:
        suffixIndex += 1 # move up to the next unit
        size = size / 1024.0 # convert the size to the next unit
    result = "%.*f %s"%(precision,size,suffixes[suffixIndex])
    return result

"""
Method to convert from GB to bytes
"""
def get_bytes_from_gigabytes(num_gb):
    to_byte = 1073741824
    return num_gb * to_byte

"""
Method to check if the target directory has enough space for a model
Raise an error if there is not enough space
"""
def check_remaining_space(num_gb, path):
    # calculate the remaining size
    num_bytes = get_bytes_from_gigabytes(num_gb) # 45 GB = # bytes
    statvfs = os.statvfs(path)
    remaining_size = statvfs.f_frsize * statvfs.f_bavail
    if remaining_size < num_bytes:
        raise SystemError("Free space in your directory: %s, space needed: %s"
    %(get_human_readable_size(remaining_size), get_human_readable_size(num_bytes)))

"""
Download the dataset triviaqa.

path: The destination path to save the compressed and uncompressed files
"""
def download_triviaqa(path):
    # check if the input path exists
    if not os.path.exists(path):
        raise Exception("The input path does not exist. Please create it before\
                        calling the method.")
    
    _URL = 'https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz'
    # get the name of the file from website
    _name = _URL.split('/')[-1]
    # remove extension
    # only get the name before the extension (cons: file.name.tar.gz = file)
    stripped_name = _name[:_name.index('.')]

    # combine the original path with the stripped name (w/o extension) of the dataset
    new_path_to_file = path + stripped_name
    if not os.path.exists(new_path_to_file):
      # create a new directory
        os.makedirs(new_path_to_file)
    else:
        print("Deleted all content in %s"%new_path_to_file)
        # delete all files in the directory
        for file in os.scandir(new_path_to_file):
            os.remove(file.path)

    # check if the directory has enough space
    check_remaining_space(num_gb=7, path=new_path_to_file)

    # retrieve and extract the compressed file
    response = requests.get(_URL, stream = True)
    # utilize tqdm to display the progress of downloading
    file_size = int(response.headers['Content-Length'])
    chunk = 1
    chunk_size=1024 * 1024
    num_bars = int(file_size / chunk_size)

    desc = "Downloading " + _name
    file_dest = new_path_to_file + '/' + _name # where to write the file to

    # Downloading the file
    with open(file_dest, 'wb') as fp:
        for chunk in tqdm(
                            response.iter_content(chunk_size=chunk_size)
                            , total= num_bars
                            , unit='MB'
                            , desc=desc
                            , leave=True # progressbar stays
                        ):
            fp.write(chunk)
    
    # Extract depending on the model
    extract_tar_gz_file(file_dest=file_dest, new_path_to_file=new_path_to_file)
    # remove the zip file
    print("Deletes the zip file at %s"%file_dest)
    os.remove(file_dest)
    # size of the generated folder
    size = get_human_readable_size(get_dir_size(new_path_to_file))
    print("\nSize of the %s directory: %s"%(new_path_to_file, size))

"""
Extract the .tar.gz compressed file using tarfile package
Uses tdqm to display the uncompress progress

file_dest: where the compressed file is
new_path_to_file: where the uncompressed files are saved
"""
def extract_tar_gz_file(file_dest, new_path_to_file):
    # open the tar.gz file
    with tarfile.open(name=file_dest) as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc='Extracting'):
            # Extract member
            tar.extract(member=member, path=new_path_to_file)

"""
Download the dataset Natural Questions.
Use the gsuutil library from the developer to download and extract the dataset

path: The destination path to save the compressed and uncompressed files
"""
def download_natural_questions(path):
    # check if the input path exists
    if not os.path.exists(path):
        raise Exception("The input path does not exist. Please create it before\
                        calling the method.")

    # combine the original path with the stripped name (w/o extension) of the dataset
    new_path_to_file = path + 'natural_questions'

    if not os.path.exists(new_path_to_file):
      # create a new directory
        os.makedirs(new_path_to_file)
    else:
        print("Deleted all content in %s"%new_path_to_file)
        # delete all files in the directory
        for file in os.scandir(new_path_to_file):
            os.remove(file.path)
    
    # check if the directory has enough space
    check_remaining_space(num_gb=45, path=new_path_to_file)

    # gsutil -m cp -R gs://natural_questions/v1.0 <path to your data directory>
    get_data = 'gsutil -m cp -R gs://natural_questions/v1.0 ' + new_path_to_file
    exit_code = os.system(get_data)
    print("`%s` ran with exit code %d" % (get_data, exit_code))

    # size of the generated folder
    size = get_human_readable_size(get_dir_size(new_path_to_file))
    print("\nSize of the %s directory: %s"%(new_path_to_file, size))