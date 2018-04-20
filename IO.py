import tarfile, pickle
import numpy as np
from IPython import embed

def load_cifar_10(path):
    tar = tarfile.open(path)
    names = sorted(tar.getnames())
    train, test = {'image': [], 'label': []}, {'image': [], 'label': []}

    for name in names:
        if 'data_batch' in name:
            data = pickle.loads(tar.extractfile(name).read(),\
                    encoding='bytes')
            train['image'].append(data[b'data'].reshape((10000, 32, 32, 3)))
            train['label'] += data[b'labels']
        if 'test' in name:
            data = pickle.loads(tar.extractfile(name).read(),\
                    encoding='bytes')
            test['image'].append(data[b'data'].reshape((10000, 32, 32, 3)))
            test['label'] += data[b'labels']

    images = np.array(train['image'])
    train['image'] = images.reshape((-1, *images.shape[2:]))
    return train, test
            
if __name__ == '__main__':
    train, test = load_cifar_10('./dataset/cifar-10-python.tar.gz')
