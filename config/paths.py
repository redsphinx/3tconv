

# general paths
saving_data = '/huge/gabras/3tconv/saving_data'
models = '/huge/gabras/3tconv/saving_data/models'
writer_path = '/huge/gabras/3tconv/saving_data/tensorboard'

# Jester
# symbolic link on godel:   /scratch/users -> /fast
jester_location = '/scratch/users/gabras/jester'
jester_data = '/scratch/users/gabras/jester/data'
jester_data_50_75 = '/scratch/users/gabras/jester/data_50_75'
jester_data_50_75_avi = '/scratch/users/gabras/jester/data_50_75_avi'
jester_data_50_75_avi_clean = '/scratch/users/gabras/jester/data_50_75_avi_clean'
jester_data_224_336 = '/scratch/users/gabras/jester/data_224_336'
jester_data_224_336_avi = '/scratch/users/gabras/jester/data_224_336_avi'
fast_jester_data_224_336_avi = '/fast/gabras/jester/data_224_336_avi'
fast_jester_data_150_224 = '/fast/gabras/jester/data_150_224'
fast_jester_data_150_224_avi = '/fast/gabras/jester/data_150_224_avi'
jester_frames = '/scratch/users/gabras/jester/frames.txt'
jester_zero = '/scratch/users/gabras/jester/zero.txt'

# UCF101
ucf101_root = '/scratch/users/gabras/ucf101/og_data'
ucf101_annotations = '/scratch/users/gabras/ucf101/og_labels'
ucf101_168_224_train = '/scratch/users/gabras/ucf101/data_168_224/train'
ucf101_168_224_test = '/scratch/users/gabras/ucf101/data_168_224/test'
ucf101_168_224_xai = '/scratch/users/gabras/ucf101/data_168_224/xai'

# dots toy dataset
dots_root = '/scratch/users/gabras/dots'
dots_dataset_avi = '/scratch/users/gabras/dots/dataset_avi'
dots_dataset_frames = '/scratch/users/gabras/dots/dataset_frames'
dots_samples = '/scratch/users/gabras/dots/samples'

# Kinetics400


# Kinetics400
# rename /fast/ to /scratch/users
kinetics400_dataset_150_224 = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224'
kinetics400_train = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224/train'
kinetics400_val = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224/valid'
kinetics400_test = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224/test'
kinetics400_train_meta = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224/train_meta_class_filelist.txt'
kinetics400_val_meta = '/scratch/users/gabras/kinetics400_downloader/dataset_150_224/valid_meta_class_filelist.txt'

# symbolic link on erdi:   /huge -> /disks/big
# symbolic link on erdi:   /scratch/users -> /disks/big

# todo: IMPORTANT: DO NOT CHANGE EXISTING PATHS. SHIT WILL BE DELETED.
# todo: IF YOU REALLY REALLY WANT TO, MAKE SURE WRITER_PATH IS VALID -> see main_file.py

