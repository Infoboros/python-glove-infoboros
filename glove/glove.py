import logging
import os
import subprocess

from .keyedvectors import KeyedVectors

logger = logging.getLogger(__name__)
dirname = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, '.tmp'), exist_ok=True)
tmppath = os.path.join(dirname, '.tmp')


class Glove:
    def __init__(
        self,
        sentences=None,
        corpus_file=None,
        vector_size=100,
        window=5,
        max_vocab=None,
        min_count=5,
        seed=42,
        workers=4,
        epochs=10
    ):
        self.clean()
        self.vector_size = vector_size
        self.window = window
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.seed = seed
        self.workers = workers
        self.epochs = epochs

        corpus_iterable = sentences
        if corpus_iterable is not None:
            corpus_iterable = '\n'.join(' '.join(sentence)
                                        for sentence in corpus_iterable)

        if corpus_iterable is not None or corpus_file is not None:
            self.build_vocab(corpus_iterable, corpus_file)
            self.cooccur(corpus_iterable, corpus_file)
            self.shuffle()
            self.train()
            self.wv = KeyedVectors.load(os.path.join(tmppath, 'vector.txt'))

    def build_vocab(self, corpus_iterable=None, corpus_file=None):
        vocab_file = os.path.join(tmppath, 'vocab.txt')
        excute = os.path.join(dirname, 'build/vocab_count')

        cli = '{} -max-vocab {} -min-count {}'.format(
            excute, self.max_vocab, self.min_count).split()
        if corpus_iterable is not None:
            result = subprocess.run(cli,
                                    input=corpus_iterable.encode(),
                                    stdout=open(vocab_file, 'w'),
                                    stderr=subprocess.PIPE)

        if corpus_file is not None:
            result = subprocess.run(cli,
                                    input=open(corpus_file, 'rb').read(),
                                    stdout=open(vocab_file, 'w'),
                                    stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def cooccur(self, corpus_iterable=None, corpus_file=None):
        cooccurrence_file = os.path.join(tmppath, 'cooccurrence.bin')
        excute = os.path.join(dirname, 'build/cooccur')

        cli = '{} -window-size {} -vocab-file {}'.format(
            excute, self.window, 'vocab.txt').split()
        if corpus_iterable is not None:
            result = subprocess.run(cli,
                                    input=corpus_iterable.encode(),
                                    stdout=open(cooccurrence_file, 'wb'),
                                    stderr=subprocess.PIPE)
        if corpus_file is not None:
            result = subprocess.run(cli,
                                    input=open(corpus_file, 'rb').read(),
                                    stdout=open(cooccurrence_file, 'wb'),
                                    stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def shuffle(self):
        cooccurrence_file = os.path.join(tmppath, 'cooccurrence.bin')
        cooccurrence_shuf_file = os.path.join(tmppath, 'cooccurrence.shuf.bin')
        excute = os.path.join(dirname, 'build/shuffle')

        cli = '{} -seed {}'.format(excute, self.seed).split()
        result = subprocess.run(cli,
                                input=open(cooccurrence_file, 'rb').read(),
                                stdout=open(cooccurrence_shuf_file, 'wb'),
                                stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def train(self):
        vocab_file = os.path.join(tmppath, 'vocab.txt')
        vector_file = os.path.join(tmppath, 'vector')
        cooccurrence_shuf_file = os.path.join(tmppath, 'cooccurrence.shuf.bin')
        excute = os.path.join(dirname, 'build/glove')

        cli = '{} -vector-size {} -threads {} -iter {} -input-file {} -vocab-file {} -save-file {} -seed {}'.format(
            excute, self.vector_size, self.workers, self.epochs, cooccurrence_shuf_file, vocab_file, vector_file, self.seed).split()

        result = subprocess.run(cli, stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def clean(self):
        for file in os.listdir(tmppath):
            os.remove(os.path.join(tmppath, file))
