import matplotlib.pyplot as plt
import numpy as np
import PIL
import utils


class Dataset:
    """
    Has all the information of the set of scenes
    - rgb_files
    - depth_files
    - label_files
    - meta_files
    _ file_names
    """

    def __init__(self, split, MAX_OBJ_ID=79, num_files_to_load=10):

        assert split in ["train", "val", "test"]

        # Assemble the dataset with the files
        file_names, rgb_files, depth_files, label_files, meta_files,  = utils.get_split_files(split)

        # rgb is not needed
        self.rgb_files = rgb_files[0:num_files_to_load]      # List
        self.depth_files = depth_files[0:num_files_to_load]  # List
        self.label_files = label_files[0:num_files_to_load]  # List
        self.meta_files = meta_files[0:num_files_to_load]    # List
        self.file_names = file_names[0:num_files_to_load]    # List
        self.number_of_files = len(self.depth_files)

        # Prepare the answer to be stored here if this dataset is predicted
        self.answer = {}
        for f in self.file_names:
            self.answer[f] = {}
            # Each element is either a None or a 2D list of size 4x4
            self.answer[f]["poses_world"] = [None] * MAX_OBJ_ID

        # Load the file information in memory
        indices = list(range(self.number_of_files))

        self.rgb_data = [np.array(PIL.Image.open(self.rgb_files[i])) for i in indices]
        print("\tloading depth data...")
        self.depth_data = [np.array(PIL.Image.open(self.depth_files[i])) / 1000 for i in indices]  # converts from mm to m
        print("\tloading label data...")
        self.label_data = [np.array(PIL.Image.open(self.label_files[i])) for i in indices]
        print("\tloading meta data...")
        self.meta_data = [utils.load_pickle(self.meta_files[i]) for i in indices]

    def get_object_ids_in_scene(self, index_scene):
        return list(self.meta_data[index_scene]['object_ids'])

    def plot_scene(self, index: int):
        rgb = self.rgb_data[index]
        depth = self.depth_data[index]
        label = self.label_data[index]
        file_name = self.file_names[index]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.subplot(1, 3, 2)
        plt.imshow(depth)
        plt.subplot(1, 3, 3)
        plt.imshow(label)  # draw colorful segmentation
        obj_ids = self.get_object_ids_in_scene(index)
        plt.suptitle(f"file: {file_name}, obj_ids: {obj_ids}")
        plt.show()

