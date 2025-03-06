import numpy as np


file_path = 'gloss_dict.npy'
gloss_dict = np.load(file_path, allow_pickle=True).item()

#print(len(gloss_dict))
print(gloss_dict)