import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv('Cleaned data.csv')
data_list = data_frame.values.tolist()

venues = ['fine restaurant', 'casual restaurant', 'street restaurant', 'other']
containers = ['bag', 'bottle', 'bowl', 'box', 'glass', 'hands-on', 'plate', 'pot', 'tray', 'other']
viewers_feeling = [1, 2, 3, 4, 5]


def cal_seconds(start_time, end_time):
    start_list = start_time.split(':')
    start_minute, start_second = int(start_list[0]), int(start_list[1])

    end_list = end_time.split(':')
    end_minute, end_second = int(end_list[0]), int(end_list[1])

    seconds = (end_minute - start_minute) * 60 + (end_second - start_second)
    return seconds


def get_duration(data_list):
    for data in data_list:
        start, end = data[0], data[1]
        seconds = cal_seconds(start, end)
        del data[0:2]
        data.insert(0, seconds)

    return data_list


def process_venues(data_list, item_list):
    for data in data_list:
        index = item_list.index(data[2])
        for i in range(len(item_list)):
            if i == index:
                data.insert(3, 1)
            else:
                data.insert(3, 0)

        del data[2]
    return data_list


def process_containers(data_list, item_list):
    for data in data_list:
        index = item_list.index(data[6])
        for i in range(len(item_list)):
            if i == index:
                data.insert(7, 1)
            else:
                data.insert(7, 0)

        del data[6]
    return data_list


def process_viewers_feelings(data_list, item_list):
    for data in data_list:
        index = item_list.index(data[16])
        for i in range(len(item_list)):
            if i == index:
                data.append(1)
            else:
                data.append(0)

        del data[16]
    return data_list


# Define values as proper (16, 1) vectors
data_list = get_duration(data_list)
data_list = process_venues(data_list, venues)
data_list = process_containers(data_list, containers)
# And plot a covariance matrix on the way
df = pd.DataFrame.from_records(data_list)
plt.matshow(df.corr())
plt.show()

data_list = process_viewers_feelings(data_list, viewers_feeling)

np_data_list = np.asanyarray(data_list)
np_data_list = np.delete(np_data_list, 0, axis=0)
np_data_list = np.delete(np_data_list, 0, axis=1)

# Standardize X, leave y out.
np_data_list = np_data_list.T
X, y = np_data_list[0:15], np_data_list[15:]
X = X.T
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Joining X and y back and save into 'Preprocess data.csv'
np_data_list = np.concatenate((X, y.T), axis=1)
data_frame = pd.DataFrame.from_records(np_data_list)
data_frame.to_csv('Preprocessed data NN.csv')

