import torch
torch.__version__           # should print 2.1.x or higher

from pinecone_text.sparse import SpladeEncoder
splade = SpladeEncoder()     # should not raise ImportError now
type(splade)                 # should be <class 'pinecone_text.sparse.splade_encoder.SpladeEncoder'>
