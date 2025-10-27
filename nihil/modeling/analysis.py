from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import typer

from nihil.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

g = nx.Graph()


@app.command()
def unweighted_analysis(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.jsonl",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    df = pd.read_json(input_path, lines=True, dtype={"id": str})
    t = 0.999
    embeddings = np.vstack(df["embeddings"].values)
    for row in df["id"]:
        g.add_node(row)
    similarity_matrix = cosine_similarity(embeddings)
    ids = df["id"].to_numpy()
    for x, i in tqdm(enumerate(similarity_matrix), total=len(similarity_matrix)):
        for y, j in enumerate(i):
            if j > t:
                if ids[x] != ids[y] and not g.has_edge(df["id"][x], df["id"][y]):
                    g.add_edge(df["id"][x], df["id"][y])
    logger.info(f"Similarity matrix:\n{similarity_matrix}")
    logger.info(f"First value: {len(similarity_matrix[0])}")
    logger.success("Process finished.")
    # Generate positions for the nodes
    pos = nx.spring_layout(g)

    # Create a Plotly figure
    fig = go.Figure()

    # Add edges to the figure
    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=data["weight"] * 5, color="gray"),
            )
        )

    # Add nodes to the figure
    for node in g.nodes():
        x, y = pos[node]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=10)))

    fig.write_html(FIGURES_DIR / "unweighted_analysis.html")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.jsonl",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    df = pd.read_json(input_path, lines=True, dtype={"id": str})
    t = 0.999

    embeddings = np.vstack(df["embeddings"].values)
    for row in df["id"]:
        g.add_node(row)
    similarity_matrix = cosine_similarity(embeddings)
    ids = df["id"].to_numpy()
    for x, i in tqdm(enumerate(similarity_matrix), total=len(similarity_matrix)):
        for y, j in enumerate(i):
            if j > t:
                if ids[x] != ids[y] and not g.has_edge(df["id"][x], df["id"][y]):
                    g.add_edge(df["id"][x], df["id"][y], weight=1)
                    g[df["id"][x]][df["id"][y]]["weight"] = 1
                elif df["id"][x] != df["id"][y] and g.has_edge(df["id"][x], df["id"][y]):
                    g[df["id"][x]][df["id"][y]]["weight"] += 1
                else:
                    continue
    pos = nx.spring_layout(g, seed=42)
    logger.info(f"Similarity matrix:\n{similarity_matrix}")
    logger.info(f"First value: {len(similarity_matrix[0])}")
    logger.success("Process finished.")
    edge_x = []
    edge_y = []

    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15, title="Node Connections", xanchor="left", titleside="right"
            ),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    fig.show()


if __name__ == "__main__":
    app()
