import pandas as pd
import sys

sys.path.append('..')
from config import cfg

def DropPedestrian(df):
    # we dont need Pedestrian here
    is_vehicle = df['Label'] != 'Pedestrian'
    df = df.loc[is_vehicle,:]
    df.reset_index(drop=True, inplace=True)
    return df
    
def DropTruck(df):
    # we dont need Pedestrian here
    is_vehicle = df['Label'] != 'Truck'
    df = df.loc[is_vehicle,:]
    df.reset_index(drop=True, inplace=True)
    return df

def SelectFrame(df, k=1):
    # select frames from the dataset
    # select one frame for every k frame
    frames = df['Frame']
    frames = [int(f[:-4]) for f in frames]
    names = list(set(frames))
    names.sort()
    select = []
    for i in range(len(names)):
        if i % k == 0:
            select.append(names[i])
    mask = [True if f in select else False for f in frames]
    df = df.loc[mask, :]
    df.reset_index(drop=True, inplace=True)
    assert len(set(df['Frame'])) == len(select)
    return df

def ProcessData():
    path_csv = 'E:\\DM\\MTCNN\\Vehicle\\CrowdAI\\labels.csv'
    df = pd.read_table(path_csv, sep=',')
    df = DropPedestrian(df)
    # df = DropTruck(df)
    df = SelectFrame(df)
    df.drop('Label', axis=1, inplace=True)
    print('frame used in crowdAI dataset is {}'.format(len(df)))
    return df
    
def DataframeToDict(df):
    mydict = {}
    for data in df.values:
        assert len(data) == 5
        frame = data[0]
        bbox = data[1:]
        if bbox[3] - bbox[1] <= 4: continue
        if bbox[2] - bbox[0] <= 4: continue

        if frame in mydict.keys():
            mydict[frame] += list(bbox)
        else:
            mydict[frame] = list(bbox)
    return mydict
    
def WriteDict(mydict, filename):
    with open(filename, 'w') as file:
        for key in mydict:
            value = mydict[key]
            info = key + ' ' + ' '.join(str(v) for v in value)
            file.write(info)
            file.write('\n')


def Run():
    df = ProcessData()
    mydict = DataframeToDict(df)
    WriteDict(mydict, cfg.path_detection_labels)
    
if __name__=='__main__':
    Run()

        
    