import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import subprocess
import pickle

class RandomTextFeatureData:
    def __init__(self):
        self.proc = subprocess.Popen([
            'python',
            'traindata_transformer.py',
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.decode()
            if line.rstrip() == 'Ready to input.':
                break
            print(line, end='')
        
        self.test_batchsize = 1

        self.max_encoderlen = 32
        self.max_decoderlen = 4 * self.max_encoderlen

    def __del__(self):
        self.proc.stdin.close()
        self.proc.kill()

    def read_data(self, batch_size, train=True):
        def load(bs, tr):
            self.proc.stdin.write(('%d,%d\n'%(bs, tr)).encode())
            self.proc.stdin.flush()

            obj = pickle.load(self.proc.stdout)
            text, encoder_input, decoder_input, decoder_trues = obj
            return text, encoder_input, decoder_input, decoder_trues

        text, encoder, decoder, decoder_true = tf.py_function(
            func=load, 
            inp=[batch_size, train], 
            Tout=[tf.string, tf.float32, tf.int32, tf.int32])
        text = tf.ensure_shape(text, [batch_size])
        encoder = tf.ensure_shape(encoder, [batch_size, self.max_encoderlen, 256])
        decoder = tf.ensure_shape(decoder, [batch_size, self.max_decoderlen])
        decoder_true = tf.ensure_shape(decoder_true, [batch_size, self.max_decoderlen])

        return {
            'text': text,
            'encoder': encoder,
            'decoder': decoder,
            'decoder_true': decoder_true,
        }

    def train_data(self, batch_size):
        def process(i):
            data = self.read_data(batch_size, True)
            data.pop('text')
            return data

        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(process)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def test_data(self, batch_size):
        def process(i):
            data = self.read_data(batch_size, False)
            data.pop('text')
            return data

        self.test_batchsize = batch_size
        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(process)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def generate_data(self):
        data = self.read_data(self.test_batchsize, False)
        return {'encoder': data['encoder'][0,...], 'text': data['text'][0].numpy().decode() }

if __name__ == '__main__':
    import tqdm

    data = RandomTextFeatureData()
    for d in data.test_data(4).take(100):
        pass

    d = data.generate_data()
    print(d)
    d = data.generate_data()
    print(d)
    d = data.generate_data()
    print(d)    
    d = data.generate_data()
    print(d)
    
    with tqdm.tqdm(total=1000) as pbar:
        for d in data.train_data(16).take(1000):
            pbar.update(1)

    for d in data.test_data(4).take(10):
        print(d)

