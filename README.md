# Content-Based Image Retrieval (CBIR) using Deep Learning

Project: Content-Based Image Retrieval using Deep features, metric learning, and efficient indexing (Faiss).

## Table of contents
- Overview
- Features
- Quick demo
- Architecture & approach
- Requirements
- Installation
- Dataset & preprocessing
- Model & training
- Indexing & retrieval (Faiss)
- Evaluation metrics
- Example usage (code snippets)
- Experiments & expected results
- Tips to improve performance
- Reproducibility
- Citation & references
- License
- Contributing

## Overview
This repository implements a Content-Based Image Retrieval (CBIR) system that uses deep convolutional networks to extract compact embeddings for images, and an efficient vector index (Faiss) to search for visually similar images at scale. The pipeline supports both supervised metric learning (triplet/contrastive loss) and transfer learning with pretrained backbones followed by an embedding head.

Use cases:
- Visual search for e-commerce or art collections
- Duplicate detection and near-duplicate search
- Instance-level retrieval (e.g., landmarks, products)
- Image clustering and browsing

## Features
- Pretrained backbone (e.g., ResNet/ResNeXt, EfficientNet) feature extractor
- Embedding head with normalization (L2) and configurable dimension (e.g., 128–1024)
- Metric learning support: triplet loss, contrastive loss, ArcFace / proxy-NCA
- Faiss-based indexing (flat, IVF, HNSW) for fast similarity search
- End-to-end training and evaluation scripts
- Evaluation metrics: mAP, precision@k, recall@k
- Example inference: index build, embedding generation, and nearest-neighbor search

## Quick demo (high-level)
1. Train or fine-tune embedding model.
2. Encode database images and build Faiss index.
3. Encode a query image and search index for top-k similar images.

## Architecture & approach
- Backbone: pretrained CNN (ResNet50/101, EfficientNet) up to global pooling layer.
- Embedding head: fully-connected layer(s) -> batchnorm -> ReLU (optional) -> L2-normalize.
- Loss:
  - Metric learning: Triplet loss with semi-hard negative mining OR contrastive loss.
  - Alternative: Classification + embedding (train as classifier then use penultimate layer).
- Indexing: Faiss for vector similarity (inner product or cosine similarity via normalized vectors).
- Retrieval: compute embedding of query -> nearest neighbor search (Top-K).

## Requirements
- Python 3.8+
- PyTorch (1.10+ recommended) or TensorFlow (if using TF variants)
- Faiss (cpu or gpu)
- torchvision, timm (optional), numpy, pandas, Pillow, scikit-learn
- tqdm, matplotlib (for visualization)

Example pip:
```bash
pip install torch torchvision faiss-cpu timm numpy pandas pillow scikit-learn tqdm matplotlib
```

## Installation
Clone repository:
```bash
git clone https://github.com/<your-org>/cbir-deep.git
cd cbir-deep
pip install -r requirements.txt
```

(If using GPU Faiss: install faiss-gpu via conda or pip following Faiss docs.)

## Dataset & preprocessing
Recommended datasets for experimentation:
- Oxford5k / Paris6k (instance-level landmarks)
- Flickr/Flickr30k / Google Landmarks
- In-shop Clothes / DeepFashion
- Custom e-commerce dataset

Preprocessing:
- Resize (shorter side to 256), center-crop or random crop to 224x224.
- Normalize using ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Data augmentation: random crop, horizontal flip, color jitter, random erasing for robust embeddings.

Example transforms (PyTorch):
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

## Model & training
Model skeleton (conceptual):
- backbone = pretrained ResNet50 (remove final fc)
- pooling = global average pooling
- embedding_head = Linear(pool_dim -> emb_dim) -> BatchNorm1d -> ReLU(optional) -> L2 normalization

Hyperparameters:
- embedding dimension: 128–512 (typical: 128 or 512)
- batch size: 64+ (depends on GPU)
- optimizer: SGD (momentum 0.9) or AdamW
- learning rate: 1e-3 (Adam) or 0.01 (SGD) with step/ cosine schedule
- epochs: 30–100 depending on dataset
- loss: triplet loss (margin 0.2) or contrastive loss

Example PyTorch training loop pseudocode:
```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        embeds = model(images)                # shape: (B, D)
        loss = triplet_loss(embeds, labels)   # or contrastive/classification
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Mining strategy:
- Batch-hard / semi-hard negative mining improves training.
- Use efficient miners from metric-learning libraries (e.g., pytorch-metric-learning).

## Indexing & retrieval (Faiss)
Faiss is recommended for large-scale nearest-neighbor retrieval.

Simple index (cosine similarity using normalized vectors -> inner product):
```python
import faiss
import numpy as np

# X: numpy array of shape (N, D) containing L2-normalized embeddings (float32)
d = X.shape[1]
index = faiss.IndexFlatIP(d)  # inner product -> cosine if vectors are normalized
index.add(X)                 # add DB embeddings
D, I = index.search(query_embedding, k=10)  # distances and indices
```

For large datasets, use IVF/HNSW:
```python
nlist = 1000
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(X)
index.add(X)
index.nprobe = 10  # adjust for recall/speed tradeoff
D, I = index.search(query_embedding, k)
```

Persistence:
```python
faiss.write_index(index, "db_index.faiss")
# load:
index = faiss.read_index("db_index.faiss")
```

## Evaluation metrics
- mAP (mean Average Precision): main metric for retrieval ranking quality.
- precision@k: fraction of relevant items in top-k.
- recall@k: fraction of relevant items recovered by top-k.
- k-NN classification accuracy (if labels available).

Example computing precision@k:
```python
# predicted_indices: (num_queries, k)
# ground_truth_lists: list of sets per query
precision_k = np.mean([len(set(pred[:k]) & gt)/k for pred, gt in zip(predicted_indices, ground_truth_lists)])
```

## Example usage
1. Train model (or download pretrained embeddings)
2. Encode images and build index:
```python
# encode function returns float32 L2-normalized vectors
db_embeddings = encode_dataset(model, db_loader)  # shape (N, D)
faiss_index = build_faiss_index(db_embeddings, index_type='IVF')
```
3. Query:
```python
query_emb = encode_image(model, "query.jpg")  # shape (1, D), L2-normalized
D, I = faiss_index.search(query_emb, k=10)
results = [db_paths[i] for i in I[0]]
```

Command-line example:
```bash
python encode.py --model checkpoints/model.pth --images data/db/ --out embeddings.npy
python build_index.py --embeddings embeddings.npy --index out/faiss.index
python query.py --index out/faiss.index --image query.jpg --k 10
```

## Experiments & expected results
- On medium-sized datasets (e.g., Oxford5k), tuned triplet-loss models with ResNet50 backbone typically achieve competitive mAP.
- For product retrieval, fine-tuning on domain-specific images gives large gains over pure off-the-shelf ImageNet features.
- IVF/HNSW indices provide good recall while keeping query latency low (<50ms for tens of millions with GPU Faiss).

## Tips to improve performance
- Use domain-specific fine-tuning (same category distribution).
- Use stronger mining strategies (batch-hard / semi-hard).
- Increase embedding dimension moderately (e.g., 512) if dataset needs more discriminative power.
- Use whitening/PCA on embeddings to reduce redundancy (but validate effect on metrics).
- Use ensemble of backbones or multi-scale features for robust similarity.
- Use HNSW for low-latency approximate nearest-neighbor on CPU.

## Reproducibility
- Fix random seeds (numpy, torch, random) and document environment (Python & package versions).
- Save model checkpoints, training logs, and evaluation scripts.
- Provide a sample script to reproduce main results (train -> encode -> build index -> eval).

## Citation & References
If you use this repository in research, please cite relevant metric learning and retrieval work:
- F. Schroff, D. Kalenichenko, J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015).
- J. Wang et al., "Learning Fine-grained Image Similarity with Deep Ranking" (2014).
- H. Jégou, M. Douze, C. Schmid, "Product quantization for nearest neighbor search" (2011).
- Faiss: https://github.com/facebookresearch/faiss

## References & useful links
- Faiss: https://github.com/facebookresearch/faiss
- pytorch-metric-learning: https://github.com/KevinMusgrave/pytorch-metric-learning
- Metric learning survey: https://arxiv.org/abs/1906.06660

## License
MIT License — see LICENSE file.

## Contributing
Contributions are welcome. Please open issues for bugs/feature requests and follow the contribution guidelines in CONTRIBUTING.md.
