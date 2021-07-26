import logging
import subprocess

logging.basicConfig(level=logging.INFO)


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

    def build_vocab(self, corpus_iterable=None, corpus_file=None):
        cli = 'build/vocab_count -max-vocab {} -min-count {}'.format(
            self.max_vocab, self.min_count).split()

        if corpus_iterable is not None:
            result = subprocess.run(cli,
                                    input=corpus_iterable.encode(),
                                    stdout=open('vocab.txt', 'w'),
                                    stderr=subprocess.PIPE)

        if corpus_file is not None:
            result = subprocess.run(cli,
                                    input=open(corpus_file, 'rb').read(),
                                    stdout=open('vocab.txt', 'w'),
                                    stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def cooccur(self, corpus_iterable=None, corpus_file=None):
        cli = 'build/cooccur -window-size {} -vocab-file {}'.format(
            self.window, 'vocab.txt').split()

        if corpus_iterable is not None:
            result = subprocess.run(cli,
                                    input=corpus_iterable.encode(),
                                    stdout=open('cooccurrence.bin', 'wb'),
                                    stderr=subprocess.PIPE)
        if corpus_file is not None:
            result = subprocess.run(cli,
                                    input=open(corpus_file, 'rb').read(),
                                    stdout=open('cooccurrence.bin', 'wb'),
                                    stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def shuffle(self):
        cli = 'build/shuffle -seed {}'.format(self.seed).split()

        result = subprocess.run(cli,
                                input=open('cooccurrence.bin', 'rb').read(),
                                stdout=open('cooccurrence.shuf.bin', 'wb'),
                                stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())

    def train(self):
        cli = 'build/glove -vector-size {} -threads {} -iter {} \
            -input-file {} -vocab-file {} -save-file {} -seed {}'.format(
            self.vector_size, self.workers, self.epochs,
            'cooccurrence.shuf.bin', 'vocab.txt', 'vector', self.seed).split()
        result = subprocess.run(cli, stderr=subprocess.PIPE)
        logging.info(result.stderr.decode().strip())


Glove(corpus_file='corpus.txt')
