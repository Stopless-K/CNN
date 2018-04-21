from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist/',one_hot=True)

class Pipe(object):
    def __init__(self, name, port):
        pass

    def send(self):
        pass

    def receive(self):
        pass


def get_train(batch_size):
    data = mnist.train.next_batch(batch_size)
    return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
            'label': data[1], 'is_training': True}
def get_val(batch_size):
    data = mnist.validation.next_batch(batch_size)
    return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
            'label': data[1], 'is_training': True}
def get_test(batch_size):
    data = mnist.test.next_batch(batch_size)
    return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
            'label': data[1], 'is_training': True}
