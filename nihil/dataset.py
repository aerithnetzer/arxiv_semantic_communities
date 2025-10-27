from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import typer

from nihil.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
tqdm.pandas()
SENTENCE_TRANSFORMER = SentenceTransformer("all-MiniLM-L6-v2")


def extract_sentences(abstract: str) -> list:
    abstract = abstract.replace("\n", " ")
    sentences = abstract.split(".")
    return sentences


def embed_sentences(sentences: list) -> np.ndarray:
    """
    Returns an array of embeddings given a list of sentences
    """
    embeddings = SENTENCE_TRANSFORMER.encode(sentences)
    return embeddings


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.jsonl",
):
    logger.info("Processing dataset...")
    df_in = pd.read_json(input_path, lines=True, dtype={"id": str})
    df_in = df_in.sample(n=5_000)
    logger.info(f"Loaded dataframe of length {len(df_in)}")
    logger.info(f"Columns of dataframe: {df_in.columns}")

    df = pd.DataFrame()
    df["abstract"] = df_in["abstract"]
    df["authors"] = df_in["authors"]

    df["title"] = df_in["title"]
    df["doi"] = df_in["doi"]
    df["id"] = df_in["id"].astype(str)
    df["update_date"] = df_in["update_date"]
    logger.info("Now extracting sentences")
    df["sentences"] = df_in["abstract"].apply(extract_sentences)
    df_exploded = df.explode("sentences", ignore_index=True)
    logger.info("Now embedding")

    # Encode each sentence
    embeddings = SENTENCE_TRANSFORMER.encode(
        df_exploded["sentences"].tolist(), show_progress_bar=True
    )

    # Add embeddings as a new column
    df_exploded["embeddings"] = embeddings.tolist()

    # Save as JSONL
    df_exploded.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    app()
