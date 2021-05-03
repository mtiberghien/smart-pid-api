from os import path, getcwd
from app.model.settings import data_dir
import sqlite3
import numpy as np

buffer_db_path = path.join(data_dir, "buffer_memory.db")


class BufferSettings:
    def __init__(self, mem_size=100000, mem_index=1):
        self.mem_size = mem_size
        self.mem_index = mem_index


class Buffer:
    def __init__(self):
        self.settings = get_buffer_settings()

    def store(self, data):
        con = sqlite3.connect(buffer_db_path)
        cursor = con.cursor()
        max_row = get_buffer_max_row(cursor)
        for d in data:
            if self.settings.mem_index > max_row:
                query = 'INSERT INTO "Data" (error, integral, derivative, saturatedIntegral, n_error, n_integral, \
                n_derivative, n_saturatedIntegral, "action", reward, done) VALUES ({});'.format(
                    ','.join([str(v) for v in d]))
                max_row += 1
            else:
                query = 'UPDATE "Data" SET error = {}, integral = {}, derivative = {}, saturatedIntegral = {},\
                        n_error = {}, n_integral = {}, n_derivative = {}, n_saturatedIntegral = {}, "action" = {},' \
                        ' reward = {}, done = {} WHERE id = {};'.format(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                                                                        d[8], d[9], d[10], self.settings.mem_index)
            cursor.execute(query)
            self.settings.mem_index = max(1, (self.settings.mem_index + 1) % (self.settings.mem_size + 1))
        con.commit()
        con.close()
        save_buffer_settings(mem_index=self.settings.mem_index)


def reset_buffer():
    con = sqlite3.connect(buffer_db_path)
    cursor = con.cursor()
    cursor.execute('DELETE FROM "Data"')
    cursor.execute('DELETE FROM SQLITE_SEQUENCE WHERE name = "Data"')
    con.commit()
    save_buffer_settings(mem_index=1)


def get_buffer_max_row(cursor):
    result = cursor.execute('SELECT seq FROM SQLITE_SEQUENCE WHERE name = "Data"').fetchone()
    return 0 if result is None else result[0]


def get_buffer_used_size():
    con = sqlite3.connect(buffer_db_path)
    cursor = con.cursor()
    return get_buffer_max_row(cursor)
    con.close()


def get_buffer_sample(batch_size):
    con = sqlite3.connect(buffer_db_path)
    cursor = con.cursor()
    max_mem = get_buffer_max_row(cursor)
    batch_size = min(batch_size, max_mem)
    batch = np.random.choice(max_mem, batch_size, replace=False) + 1
    where_clause = 'WHERE id IN ({})'.format(",".join([str(num) for num in batch]))
    rng = np.random.default_rng()
    query = 'SELECT error, integral, derivative, saturatedIntegral, n_error, n_integral, n_derivative,  \
    n_saturatedIntegral, "action", reward, done FROM "Data" {}'.format(where_clause)
    data = np.array(cursor.execute(query).fetchall())
    rng.shuffle(data)
    con.close()
    if data.size > 0:
        states = data[:, 0:4]
        new_states = data[:, 4:8]
        actions = data[:, 8:9]
        rewards = data[:, 9:10]
        are_done = data[:, 10:11]
    else:
        states = np.array([])
        actions = np.array([])
        rewards = np.array([])
        new_states = np.array([])
        are_done = np.array([])
    return batch_size, states, actions, rewards, new_states, are_done


def save_buffer_settings(**kwargs):
    settings = get_buffer_settings()
    if 'mem_size' in kwargs:
        settings.mem_size = kwargs.get('mem_size')
    if 'mem_index' in kwargs:
        settings.mem_index = kwargs.get('mem_index')
    con = sqlite3.connect(buffer_db_path)
    cursor = con.cursor()
    settings.mem_index = max(1, min(settings.mem_size, settings.mem_index))
    query = 'UPDATE Settings SET max_size = {}, current_index = {};'.format(settings.mem_size,
                                                                            settings.mem_index)
    cursor.execute(query)
    con.commit()
    con.close()


def get_buffer_settings():
    if path.exists(buffer_db_path):
        con = sqlite3.connect(buffer_db_path)
        cursor = con.cursor()
        mem_size, mem_index = cursor.execute('SELECT max_size, current_index FROM Settings LIMIT 1;').fetchone()
        return BufferSettings(mem_size, mem_index)
    return BufferSettings()
