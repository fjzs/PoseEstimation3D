import paths
import pickle
import os


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_split_files(split_name):
    """
    Returns the rgb, depth, label and meta files as lists
    :param split_name:
    :return:
    """
    assert split_name in ["train", "val", "test"]

    splits_dir = paths.SPLITS
    files_dir = paths.TRAINING_DIR
    if split_name == "test":
        files_dir = paths.TESTING_DIR

    with open(os.path.join(splits_dir, f"{split_name}.txt"), 'r') as f:

        file_names = [line.strip() for line in f if line.strip()]
        prefix = [os.path.join(files_dir, line) for line in file_names]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]

    return file_names, rgb, depth, label, meta
