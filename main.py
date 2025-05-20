import requests
import os
import json
from pathlib import Path
from base64 import decodebytes
import base64
import numpy as np
import io
import logging

# Configure logging to show INFO level messages to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    generate_dna_sequence()

    embeddings = get_embeddings()

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


def generate_dna_sequence() -> list[np.ndarray]:
    key = os.getenv("NVIDIA_NIM_API_KEY")
    if not key:
        raise ValueError("NVIDIA_NIM_API_KEY environment variable not set. Please set it before running the script.")

    logging.info("Generating DNA sequence from API...")
    r = requests.post(
    url=os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"),
    headers={"Authorization": f"Bearer {key}"},
    json={
        "sequence": "ACTGACTGACTGACTG",
        "num_tokens": 8,
        "top_k": 1,
        "enable_sampled_probs": True,
    },
)

    if "application/json" in r.headers.get("Content-Type", ""):
        print(r, "Saving to output.json:\n", r.text[:200], "...")
        Path("output.json").write_text(r.text)
    elif "application/zip" in r.headers.get("Content-Type", ""):
        print(r, "Saving large response to data.zip")
        Path("data.zip").write_bytes(r.content)
    else:
        print(r, r.headers, r.content)
    
def get_embeddings() -> list[np.ndarray]:
    key = os.getenv("NVIDIA_NIM_API_KEY")
    if not key:
        raise ValueError("NVIDIA_NIM_API_KEY environment variable not set. Please set it before running the script.")

    logging.info("Getting embeddings from API...")
    r = requests.post(
    url=os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"),
    headers={"Authorization": f"Bearer {key}"},
    json={
        "sequence": "ACTGACTGACTGACTG",
        "output_layers": ['embedding_layer'] # Token Embeddings should use a middle layer (e.g. Layer 26): https://github.com/ArcInstitute/evo2/issues/95#issuecomment-2859371381
    },
)

    if "application/json" in r.headers.get("Content-Type", ""):
        print(r, "Saving to output.json:\n", r.text[:200], "...")
        Path("output.json").write_text(r.text)
    elif "application/zip" in r.headers.get("Content-Type", ""):
        print(r, "Saving large response to data.zip")
        Path("data.zip").write_bytes(r.content)
    else:
        print(r, r.headers, r.content)
    
    logging.info("Embeddings stored")


if __name__ == "__main__":
    main()
