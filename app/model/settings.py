from os import path

root_dir = 'app'
data_dir = 'data'
db_file = "buffer_memory.db"
settings_file = "agent.json"
buffer_db_path = path.join(root_dir, data_dir, db_file)
settings_file_path = path.join(root_dir, data_dir, settings_file)


def get_data_dir():
    global root_dir, data_dir
    return path.join(root_dir, data_dir)


def get_buffer_db_path():
    global buffer_db_path
    return buffer_db_path


def get_settings_file_path():
    global settings_file_path
    return settings_file_path


def init_dirs(root_path):
    global root_dir
    global buffer_db_path
    global settings_file_path
    root_dir = root_path
    buffer_db_path = path.join(root_path, data_dir, db_file)
    settings_file_path = path.join(root_dir, data_dir, settings_file)



