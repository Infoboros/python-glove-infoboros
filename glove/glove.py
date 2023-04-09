import os
import subprocess

from .keyedvectors import KeyedVectors

dirname = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, '.tmp'), exist_ok=True)
tmppath = os.path.join(dirname, '.tmp')
log = os.path.join(tmppath, 'train.log')


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
            epochs=10,
            verbose=False,
    ):
        self.clean()
        self.vector_size = vector_size
        self.window = window
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.seed = seed
        self.workers = workers
        self.epochs = epochs
        self.verbose = verbose

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
        log_pipe = self.get_log_pipe()

        if corpus_iterable is not None:
            subprocess.run(cli,
                           input=corpus_iterable.encode(),
                           stdout=open(vocab_file, 'w'),
                           stderr=log_pipe)

        if corpus_file is not None:
            subprocess.run(cli,
                           input=open(corpus_file, 'rb').read(),
                           stdout=open(vocab_file, 'w'),
                           stderr=log_pipe)

    def cooccur(self, corpus_iterable=None, corpus_file=None):
        vocab_file = os.path.join(tmppath, 'vocab.txt')
        cooccurrence_file = os.path.join(tmppath, 'cooccurrence.bin')
        excute = os.path.join(dirname, 'build/cooccur')

        cli = '{} -window-size {} -vocab-file {}'.format(
            excute, self.window, vocab_file).split()

        log_pipe = self.get_log_pipe()
        if corpus_iterable is not None:
            subprocess.run(cli,
                           input=corpus_iterable.encode(),
                           stdout=open(cooccurrence_file, 'wb'),
                           stderr=log_pipe)
        if corpus_file is not None:
            subprocess.run(cli,
                           input=open(corpus_file, 'rb').read(),
                           stdout=open(cooccurrence_file, 'wb'),
                           stderr=log_pipe)
        self.print_log()

    def shuffle(self):
        cooccurrence_file = os.path.join(tmppath, 'cooccurrence.bin')
        cooccurrence_shuf_file = os.path.join(tmppath, 'cooccurrence.shuf.bin')
        excute = os.path.join(dirname, 'build/shuffle')

        cli = '{} -seed {}'.format(excute, self.seed).split()
        subprocess.run(cli,
                       input=open(cooccurrence_file, 'rb').read(),
                       stdout=open(cooccurrence_shuf_file, 'wb'),
                       stderr=self.get_log_pipe())
        self.print_log()

    def train(self):
        vocab_file = os.path.join(tmppath, 'vocab.txt')
        vector_file = os.path.join(tmppath, 'vector')
        cooccurrence_shuf_file = os.path.join(tmppath, 'cooccurrence.shuf.bin')
        excute = os.path.join(dirname, 'build/glove')

        cli = '{} -vector-size {} -threads {} -iter {} -input-file {} -vocab-file {} -save-file {} -seed {}'.format(
            excute,
            self.vector_size,
            self.workers, self.epochs,
            cooccurrence_shuf_file,
            vocab_file,
            vector_file,
            self.seed
        ).split()

        subprocess.run(
            cli,
            stderr=self.get_log_pipe()
        )
        self.print_log()

    def get_log_pipe(self):
        return open(log, 'wb')

    def print_log(self):
        if self.verbose:
            with open(log) as f:
                [print(line) for line in f.read().split('\n')]

    def clean(self):
        return
        for file in os.listdir(tmppath):
            os.remove(os.path.join(tmppath, file))
