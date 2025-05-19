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
    embeddings = load_embeddings_from_api()

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


def load_embeddings_from_api() -> list[np.ndarray]:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    key = os.getenv("NVIDIA_NIM_API_KEY")
    if not key:
        raise ValueError("NVIDIA_NIM_API_KEY environment variable not set. Please set it before running the script.")

    logging.info("Loading embeddings from API...")
    r = requests.post(
        url=os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"),
        headers={"Authorization": f"Bearer {key}"},
        json={
            "sequence": "ACTGTCGATGCATCA",
            # embedding_layer, sequential.21.mlp.l3 doesn't work
            "output_layers": ['embedding_layer'] # Token Embeddings should use a middle layer (e.g. Layer 26): https://github.com/ArcInstitute/evo2/issues/95#issuecomment-2859371381
        },
    )

    logging.info(r.text)

    data = json.loads(r.text)
    decoded_data = base64.b64decode(data['data'].encode("ascii"))
    embeddings = np.load(io.BytesIO(decoded_data))['embedding_layer.output']

    logging.info(np.all(embeddings[0,0] == embeddings[0,14]))  # Output: True

if __name__ == "__main__":
    main()
