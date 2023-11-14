import os
import argparse
import tsnecuda
import pickle

def parse_option():
    parser = argparse.ArgumentParser('argument for t-SNE embedding')

    parser.add_argument('--path', type=str, default=None, help='path to pickeled feature embedding')

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    path = opt.path

    print("Training data")
    print("load feature embedding")
    with open(os.path.join(path, "embedding_train"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding = entry['data']

    print("compute t-SNE embedding")
    embedding_tSNE = tsnecuda.TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(embedding)

    print("writ t-SNE embedding")
    entry['data'] = embedding_tSNE
    with open(os.path.join(path, "embedding_tSNE_train"), 'wb') as f:
        pickle.dump(entry, f, protocol=-1)

    print("Test data")
    print("load feature embedding")
    with open(os.path.join(path, "embedding_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding = entry['data']

    print("compute t-SNE embedding")
    embedding_tSNE = tsnecuda.TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(embedding)

    print("writ t-SNE embedding")
    entry['data'] = embedding_tSNE
    with open(os.path.join(path, "embedding_tSNE_test"), 'wb') as f:
        pickle.dump(entry, f, protocol=-1)


if __name__ == '__main__':
    main()