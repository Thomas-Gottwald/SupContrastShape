# deactivate
# source /home/tgottwald/tsnecuda_venv/bin/activate
# export LD_LIBRARY_PATH=/home/tgottwald/tsnecuda_venv/lib64
# CUDA_VISIBLE_DEVICES=0 python tsnecuda_embedding.py
import os
import tsnecuda
import numpy as np
import pickle

os.environ['LD_LIBRARY_PATH']='/home/tgottwald/tsnecuda_venv/lib64'

# tsnecuda.test()

print("load feature embedding")
# with open("./save/embeddings/tSNEcuda/embedding_feature", 'rb') as f:
with open("embedding_test", 'rb') as f:
    embedding = pickle.load(f, encoding='latin1')

print("compute t-SNE embedding")
embedding_tSNE = tsnecuda.TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(embedding)

print("writ t-SNE embedding")
# with open("./save/embeddings/tSNEcuda/embedding_tSNE", 'wb') as f:
with open("embedding_test_tSNE", 'wb') as f:
    pickle.dump(embedding_tSNE, f, protocol=-1)
