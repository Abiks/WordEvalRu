from pathlib import Path
import os

from tqdm import tqdm
from sklearn_crfsuite import CRF, metrics


PATH_TO_DATA = '../../data/fact-ru-eval/'


class CheckerNERFactRuEval:
    def __init__(self, embedder, label_transform=None):
        self.embedder = embedder
        self.label_transform = label_transform
        
    def get_token_features_from_embedding(self, token_embedding):
        token_features = dict()

        for i, e in enumerate(token_embedding):
            token_features[f'emb_{i}'] = e
        return token_features


    def sent2features(self, sentence):
        # I'm using crfsuite, this module requires word features as list of dicts. 
        features = []  # TODO: rename this variable
        sentence_embeddings = self.embedder.get_word_embeddings(sentence)
        for token_embedding in sentence_embeddings:
            features.append(self.get_token_features_from_embedding(token_embedding))

        return features

    def read_conll(self, path_to_file):
        data_texts = []
        data_labels = []

        with path_to_file.open() as f:
            data = f.read().split('\n\n')[1:]

        for sample in tqdm(data):
            splitted_sample = sample.split('\n')
            texts = []
            labels = []

            for line in splitted_sample:
                line_splitted = line.split()
                if line_splitted:
                    texts.append(line_splitted[0])
                    label = line_splitted[2]
                    if self.label_transform:
                        label = labels_transform.get(label, label)
                    labels.append(label)

            text_features = self.sent2features(texts)
            data_texts.append(text_features)
            data_labels.append(labels)

        return data_texts, data_labels  


    def check(self):
        path_to_data = Path(__file__).resolve().parent / PATH_TO_DATA
        X_train, y_train = self.read_conll(path_to_data / 'train.txt')
        X_test, y_test = self.read_conll(path_to_data / 'test.txt')
        crf = CRF(  # TODO: add GridSearch
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        labels = list(crf.classes_)
        labels.remove('O')

        y_pred = crf.predict(X_test)
        
        # TODO: add more metrics
        f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        
        return f1