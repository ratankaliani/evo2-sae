from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from evo2 import Evo2

def main():
    embeddings, labels = load_genomic_sequence()
    plot_umap(embeddings, labels)

def load_genomic_sequence() -> tuple[list[np.ndarray], list[str]]:
    print("Loading model...")
    # 7B‑param Evo 2 (smallest)
    model = AutoModel.from_pretrained("arcinstitute/savanna-evo2-7b-sae")
    tokenizer = AutoTokenizer.from_pretrained("arcinstitute/evo2-7b-sae")

    print("Fetching genomic sequence...")

    df = pd.read_csv("clinvar_subset.tsv", sep="\t")
    embeddings = []
    labels = []
    for i, row in df.iterrows():
        seq = fetch_genomic_sequence(row.chrom, row.pos-50, row.pos+50)  # ±50 bp window
        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            emb = model.sae_head(**inputs).last_hidden_state.mean(dim=1)  # pooled vector
        embeddings.append(emb.squeeze().cpu().numpy())
        labels.append(row.clinical_significance)
        print(f"Processed {i+1}/{len(df)}")
    return embeddings, labels

def plot_umap(embeddings: list[np.ndarray], labels: list[str]):
    reducer = umap.UMAP(n_components=2)
    pts = reducer.fit_transform(embeddings)

    plt.scatter(
        pts[:,0], pts[:,1],
        c=[1 if lab=="pathogenic" else 0 for lab in labels],
        cmap="coolwarm", alpha=0.7
    )
    plt.legend(["Benign","Pathogenic"])
    plt.title("Evo 2 SAE Embeddings UMAP")
    plt.show()

if __name__ == "__main__":
    main()
