import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import HfArgumentParser
from util import util

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        metadata={"help": "Which dataset 'newsqa', 'nq', 'quac', 'triviaqa' dataset"}, 
    )
    dataset_path: Optional[str] = field(
        default="~/.cache/conversion/datasets",
        metadata={"help": "Path to the source file to be downloaded"}, 
    )
    to_file_path: Optional[str] = field(
        default="./converted_files",
        metadata={"help": "Path for generated json file(s)"}, 
    )

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    
if __name__ == "__main__":
    main()