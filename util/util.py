import json
from urllib import response
import pandas as pd
import os
import spacy

import requests
import tarfile
from tqdm import tqdm



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