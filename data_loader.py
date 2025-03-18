import os
import cv2

class DataLoader:
    def __init__(self, dataset_path, cv2_flags=cv2.IMREAD_GRAYSCALE):
        free_folder_path = dataset_path + "/free/"
        infested_folder_path = dataset_path + "/infested/"
        free_file_list = os.listdir(free_folder_path)
        infested_file_list = os.listdir(infested_folder_path)

        complete_path_free_file_list = [free_folder_path + s for s in free_file_list]
        complete_path_infested_file_list = [infested_folder_path + s for s in infested_file_list]

        y_free = [0 for _ in range(len(free_file_list))]
        y_infested = [1 for _ in range(len(infested_file_list))]

        self.image_files = complete_path_free_file_list + complete_path_infested_file_list
        self.labels = y_free + y_infested

        self.cv2_flags = cv2_flags
        
        self.index = 0

       
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.index >= len(self.image_files):
            raise StopIteration
        # image = cv2.imread(self.image_files[self.index], self.cv2_flags)
        image = self.read_image()
        label = self.labels[self.index]
        self.index += 1
        return image, label


    def read_image(self):
        image = cv2.imread(self.image_files[self.index], self.cv2_flags)
        if image is None:
            raise ValueError("Error loading image. Check the file path.")
        return image
