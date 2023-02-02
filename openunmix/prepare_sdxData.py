from pathlib import Path
import random
import shutil
from os import path
import argparse

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

def prepare_dataset(
    input,
    output="output",
    seed=1337,
    ratio=(0.8,0.1,0.1),
    move=False
):
    if not round(sum(ratio),5) == 1:
        raise ValueError("The sum of ratio (train,valid,test) is more than 1")
    if not len(ratio) in (2,3):
        raise ValueError("'ratio' should be (train,test) or (train,valid,test)")

    check_input_format(input)

    if use_tqdm:
        prog_bar = tqdm(desc=f"Copying folders",unit=" folders")
        
    split_class_dir_ratio(
            input,
            output,
            ratio,
            seed,
            prog_bar if use_tqdm else None,
            move,
        )
    
    if use_tqdm:
        prog_bar.close()

def check_input_format(input):
    p_input = Path(input)
    if not p_input.exists():
        err_msg = f'The input folder "{input}" does not exist'
        if not p_input.is_absolute():
            err_msg += f' Your relative path cannot be found from the current working directory "{Path.cwd()}".'
        raise ValueError(err_msg)

    if not p_input.is_dir():
        raise ValueError(f'The provided input folder "{input}" is not a directory')

    dirs = list_folders(input)
    if len(dirs) == 0:
        raise ValueError(
            f'The input data is not in a right format. Within your folder "{input}" there are no directories.'
        )

def setup_folders(data_dir,seed):
    random.seed(seed)

    folders = list_folders(data_dir)
    folders.sort()
    random.shuffle(folders)
    return folders

def split_class_dir_ratio(data_dir, output, ratio, seed, prog_bar, move):
    folders = setup_folders(data_dir, seed)

    split_train_idx = int(ratio[0]*len(folders))
    split_val_idx = split_train_idx+int(ratio[1]*len(folders))

    splits = split_folders(folders, split_train_idx, split_val_idx, len(ratio)==3)
    copy_folders(splits, output, prog_bar, move)

def split_folders(folders, split_train_idx, split_val_idx, use_test):
    folders_train = folders[:split_train_idx]
    folders_val = (folders[split_train_idx:split_val_idx] if use_test else folders[split_train_idx:])

    split = [(folders_train,"train"),(folders_val,"valid")]
    if use_test:
        folders_test = folders[split_val_idx:]
        split.append((folders_test,"test"))
    return split

# def split_files

def list_folders(directory):
    """
    return all subdirectories in a directory
    """
    return [f for f in Path(directory).iterdir() if f.is_dir()]

def list_files(directory):
    return [f for f in Path(directory).iterdir() if f.is_file() and not f.name.startswith(".")]


def copy_folders(folders_type, output, prog_bar, move):
    copy_fun = shutil.move if move else shutil.copytree

    for (folders, folder_type) in folders_type:
        for folder in folders:
            class_name = path.split(folder)[1]
            full_path = path.join(output,folder_type, class_name)
            copy_fun(str(folder),str(full_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preparation for SDX 2023")
    parser.add_argument(
        "--root",
        type=str,
        help="root path of dataset",
        )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory",
        default = "output"
    )
    parser.add_argument("--seed", type=int, default=1321)
    parser.add_argument(
        "--ratio",
        type = int,
        nargs='+',
        default=[0.8,0.1,0.1],
        help="Ratio of split (default is 0.8, 0.1, 0.1)"
    )
    parser.add_argument(
        "--move", action="store_true", help=("Move data to output folder")
    )

    args = parser.parse_args()

    prepare_dataset(args.root,args.output,args.seed,args.ratio,args.move)