import pickle

import click
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier


@click.command()
@click.option("-m", type=str, required=True, help="Path to a trained model.")
@click.option("-t", type=str, required=True, help="Text to make a prediction on.")
def pred(m: str, t: str) -> None:
    # Load the same sentence-transformer as the train time to extract feature
    encoder = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")

    # Load trained model
    with open(m, "rb") as fp:
        clf: MLPClassifier = pickle.load(fp)

    # Get embeddings for the input text.
    embs = encoder.encode([t], convert_to_numpy=True)

    # Predict!
    print(clf.predict(embs)[0])
