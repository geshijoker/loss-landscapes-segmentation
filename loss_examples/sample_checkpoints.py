import os

path = "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049"
dir_list = os.listdir(path) 

freq = 5

files = []
for file in dir_list:
    if not file.endswith('.pt'):
        continue
    iteration = int(file.split('-')[0].split('iter')[1])
    if iteration%freq == 0:
        files.append("\"" + os.path.join(path, file) + "\"")
print(" ".join(files))