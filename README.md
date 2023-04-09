# python-glove

Python implementation of [GloVe](https://nlp.stanford.edu/projects/glove/)

GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
Training is performed on aggregated global word-word co-occurrence statistics from a corpus,
and the resulting representations showcase interesting linear substructures of the word vector space.

## Installation

### From source:
```bash
$ git clone https://github.com/Infoboros/python-glove-infoboros.git
$ cd python-glove
$ python setup.py install
```

## Usage
```python
>>> from glove import Glove

>>> model = Glove(corpus_file='test/corpus.txt', vector_size=100, window=5, min_count=5, epochs=10, verbose=True)
BUILDING VOCABULARY
Processed 3381866 tokens.
Counted 24746 unique words.
Truncating vocabulary at min count 5.
Using vocabulary of size 7731.

COUNTING COOCCURRENCES
window size: 5
context: symmetric
max product: 10485784
overflow length: 28521267
Reading vocab from file "/Users/hieunguyen/Desktop/ICON/python-glove/glove/.tmp/vocab.txt"...loaded 7731 words.
Building lookup table...table contains 28729425 elements.
Processed 3381866 tokens.
Writing cooccurrences to disk.......2 files in total.
Merging cooccurrence files: processed 2680220 lines.

Using random seed 42
SHUFFLING COOCCURRENCES
array size: 127506841
Shuffling by chunks: processed 2680220 lines.
Wrote 1 temporary file(s).
Merging temp files: processed 2680220 lines.

TRAINING MODEL
Read 2680220 lines.
Initializing parameters...Using random seed 42
done.
vector size: 100
vocab size: 7731
x_max: 100.000000
alpha: 0.750000
07/27/21 - 10:33.06AM, iter: 001, cost: 0.061383
07/27/21 - 10:33.08AM, iter: 002, cost: 0.044241
07/27/21 - 10:33.11AM, iter: 003, cost: 0.039158
07/27/21 - 10:33.13AM, iter: 004, cost: 0.036379
07/27/21 - 10:33.15AM, iter: 005, cost: 0.033546
07/27/21 - 10:33.19AM, iter: 006, cost: 0.030526
07/27/21 - 10:33.21AM, iter: 007, cost: 0.027430
07/27/21 - 10:33.23AM, iter: 008, cost: 0.024350
07/27/21 - 10:33.25AM, iter: 009, cost: 0.021456
07/27/21 - 10:33.27AM, iter: 010, cost: 0.019001
>>> print(model.wv.vectors.shape)
(7732, 100)
>>> print(model.wv.vectors)
[[-0.7832    0.230984  0.328523 ... -0.938997 -0.772137  0.827372]
 [ 0.119143  0.06323   0.773245 ... -0.802186 -1.225709  0.65204 ]
 [-0.382861 -0.607985  0.218486 ... -0.402255 -1.133209  0.395143]
 ...
 [-0.026736  0.005838 -0.052565 ...  0.016259  0.022208 -0.015785]
 [-0.017614  0.020005 -0.055972 ...  0.024249  0.039124 -0.055554]
 [-0.012019  0.008404 -0.034215 ...  0.026566  0.037037 -0.031336]]
>>> print(model.wv.index_to_key[:10])
[',', '.', 'là', 'tôi', 'một', 'có', 'và', 'những', 'chúng', 'của']
>>> print(len(model.wv))
7732
```

## Development

Pull requests are welcome.
Fun hacking!