import pickle

import click
import sienna
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier


@click.command()
@click.option(
    "-f", type=str, required=True, help="Path to jsonl file contains training samples."
)
@click.option("-o", type=str, required=True, help="Path to save trained model.")
def train(f: str, o: str) -> None:
    # Load jsonl file as a List[Dict]
    data = sienna.load(f)

    # Extract texts and labels for supervised learning.
    texts, labels = zip(*[(x["text"], x["label"]) for x in data])

    # Load sentence-transformer model.
    encoder = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")

    # Get embeddings for each of text.
    embs = encoder.encode(texts, convert_to_numpy=True)

    # Train MLP classifier
    clf = MLPClassifier(random_state=1, max_iter=50).fit(embs, labels)

    # Save trained model as a pickle file
    with open(o, "wb") as fp:
        pickle.dump(clf, fp)
