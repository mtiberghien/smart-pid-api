from os import path

root_dir = 'app'
data_dir = path.join(root_dir, 'data')


def init_dirs(root_path):
    global root_dir
    global data_dir
    root_dir = root_path
    data_dir = path.join(root_dir, 'data')



