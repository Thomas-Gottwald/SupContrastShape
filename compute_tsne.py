import os
import argparse
import numpy as np
import tsnecuda
import pickle

def parse_option():
    parser = argparse.ArgumentParser('argument for t-SNE embedding')

    parser.add_argument('--path', type=str, default=None, help='path to pickeled feature embedding')
    parser.add_argument('--path_second', type=str, default=None, help='path to a second pickeled feature embedding')
    parser.add_argument('--path_save', type=str, default=None, help='path to save the t-SNE embedding. If not given they will be saved at path.')

    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'innerproduct'], help='choose metric')

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    path = opt.path
    path_second = opt.path_second
    path_save = opt.path_save if opt.path_save else path

    metric = opt.metric

    for split in ["train", "test"]:
        print(f"Data split {split}")
        if os.path.isfile(os.path.join(path, f"embedding_{split}")):
            print("load feature embedding")
            with open(os.path.join(path, f"embedding_{split}"), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                embedding = entry['data']

            if path_second:
                print("load second feature embedding")
                with open(os.path.join(path_second, f"embedding_{split}"), 'rb') as f:
                    entry_second = pickle.load(f, encoding='latin1')
                    embedding = np.append(embedding, entry_second['data'], axis=0)
                    entry['labels'] = np.append(entry['labels'], entry_second['labels'])

            print("compute t-SNE embedding")
            embedding_tSNE = tsnecuda.TSNE(n_components=2, perplexity=30, learning_rate=10, metric=metric).fit_transform(embedding)

            print("writ t-SNE embedding")
            entry['data'] = embedding_tSNE
            with open(os.path.join(path_save, f"embedding_tSNE_{split}"), 'wb') as f:
                pickle.dump(entry, f, protocol=-1)
        else:
            print(f"No {split} data not found!")


if __name__ == '__main__':
    main()