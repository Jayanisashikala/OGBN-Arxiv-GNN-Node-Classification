# OGBN-Arxiv Graph Neural Network Node Classification

This project uses Graph Neural Networks (GNNs) to predict the subject category of research papers, using the **OGBN-Arxiv citation network**.

The goal is simple: given a paper's text-based features and its citation links to other papers, predict which of 40 computer-science subject areas it belongs to.

## Dataset

We use the [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset from the Open Graph Benchmark.

| Property | Value |
|---|---:|
| Papers (nodes) | 169,343 |
| Citations (directed edges) | 1,166,243 |
| Features per paper | 128 |
| Subject categories (classes) | 40 |
| Task | Multi-class node classification |
| Data split | Official OGB chronological split |

Each paper is a node. A directed edge from paper A to paper B means paper A cites paper B. The 128 features are numerical embeddings generated from each paper's title and abstract.

## What the Project Does

### Main tasks

1. **Tensor Basics** – Core PyTorch tensor operations: creation, indexing, reshaping, matrix multiplication, broadcasting, aggregation, and GPU usage.
2. **Graph Analysis** – Builds an edge list, studies node features, and analyzes degree distribution, graph density, and connected components. Includes a sample subgraph visualization.
3. **Data Preparation** – Uses the official train/validation/test splits, checks for missing values, and adds reverse edges so information can flow in both directions during message passing.
4. **GNN Models** – Builds two models, GCN and GraphSAGE, each with 3 graph layers, batch normalization, ReLU activation, dropout, and a residual connection.
5. **Training** – Trains both models using cross-entropy loss (with label smoothing), the Adam optimizer, a learning-rate scheduler, gradient clipping, and early stopping. Compares two hyperparameter configurations for each model.
6. **Evaluation** – Measures accuracy, macro precision, macro recall, and macro F1 on both validation and test sets, plus confusion matrices.
7. **Explainability** – Uses PCA to visualize learned embeddings, and checks whether a paper's predicted class agrees with its neighbours' classes.
8. **Dashboard** – A Streamlit app showing graph statistics, model performance, and predictions.

### Bonus tasks

9. **Graph Transformer** – Attention-based model tested on a 10,000-node sample.
10. **GATv2** – A more advanced attention-based GNN, also tested on the sample.
11. **Knowledge Graph Extension** – A small graph connecting papers to their subjects.
12. **Self-Supervised Learning** – Trains an encoder to reconstruct masked features without using labels, then tests how useful the learned embeddings are.

## Data Split

| Split | Papers | Percentage |
|---|---:|---:|
| Training | 90,941 | 53.70% |
| Validation | 29,799 | 17.60% |
| Test | 48,603 | 28.70% |

These groups don't overlap. Feature statistics are calculated using only the training set to avoid data leakage. The original OGB features are kept as-is. Reverse edges are added only for message passing, bringing the total edge count used by the models to 2,315,598.

## The Two Models

**GCN (Graph Convolutional Network)** – Spreads and averages information across connected papers using normalized graph convolution.

**GraphSAGE** – Combines each paper's own features with information gathered from its neighbours, rather than just averaging everything together.

Both models share the same basic structure:
- 3 graph message-passing layers
- 2 batch-normalization layers
- ReLU activation
- Dropout
- A residual connection
- A final layer producing scores for the 40 classes

## Training Setup

Two hyperparameter configurations were tested for each model:

| Configuration | Hidden size | Learning rate | Dropout | Max epochs | Patience |
|---|---:|---:|---:|---:|---:|
| 1 | 128 | 0.010 | 0.5 | 150 | 20 |
| 2 | 256 | 0.005 | 0.4 | 150 | 20 |

Training used:
- Cross-entropy loss with label smoothing (0.05)
- Adam optimizer with weight decay (5e-4)
- `ReduceLROnPlateau` learning-rate scheduler
- Gradient clipping (max norm 2.0)
- Early stopping based on validation accuracy

Configuration 2 (256 hidden units, learning rate 0.005, dropout 0.4) gave the best results for both models.

## Results

### Main model comparison

| Model | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| GCN | Validation | 71.72% | 60.62% | 46.02% | 49.13% |
| GCN | Test | 70.46% | 57.32% | 44.08% | 47.23% |
| GraphSAGE | Validation | 72.63% | 60.05% | 49.49% | 51.85% |
| GraphSAGE | Test | **71.03%** | **60.11%** | **47.64%** | **49.69%** |

GraphSAGE performed slightly better than GCN on every test metric. Both models score lower on macro precision/recall/F1 than on accuracy because the 40 subject categories are imbalanced (some categories have far more papers than others).

### Explainability examples

| Node | True Class | Predicted Class | Confidence | Correct? | Neighbours Agreeing |
|---:|---:|---:|---:|---|---:|
| 451 | 24 | 24 | 69.92% | Yes | 81.82% |
| 346 | 24 | 10 | 43.54% | No | 0.00% |

PCA gives an overall picture of how papers cluster in the embedding space. Neighbourhood analysis checks whether a paper's prediction matches what its cited/citing papers are labeled as.

### Bonus results

| Method | Tested On | Validation Accuracy | Test Accuracy |
|---|---|---:|---:|
| Graph Transformer | 10,000-node sample | 50.60% | 49.30% |
| GATv2 | 10,000-node sample | 52.85% | 51.15% |
| Knowledge Graph | 12 sample papers | — | Triple table + visualization |
| Self-Supervised Encoder | 10,000-node sample | 16.90% | 15.40% |

The bonus attention models (Graph Transformer, GATv2) were tested on a smaller sample, so their numbers aren't directly comparable to the main GCN/GraphSAGE results. During self-supervised pretraining, the reconstruction loss dropped from 0.0580 to 0.0138, showing the encoder was learning useful patterns even without labels.

## Tools Used

Python, PyTorch, PyTorch Geometric, OGB, NumPy, pandas, SciPy, scikit-learn, NetworkX, Matplotlib, seaborn, Streamlit, Google Colab

## How to Run

### Clone the project

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Install requirements

```bash
pip install torch torch-geometric ogb streamlit pandas numpy scipy scikit-learn networkx matplotlib seaborn
```

A GPU is recommended for training the full-graph models.

### Run the notebooks

The project is split into 12 notebooks inside the `notebooks/` folder, one per task. Run them **in order**, since later notebooks depend on files saved by earlier ones:

1. `Task_01_-_Tensor_Fundamentals.ipynb`
2. `Task_02_-_Graph_Representation.ipynb`
3. `Task_03_-_Graph_Data_Preparation.ipynb`
4. `Task_04_-_Graph_Neural_Network.ipynb`
5. `Task_05_-_Model_Training.ipynb`
6. `Task_06_-_Model_Evaluation.ipynb`
7. `Task_07_-_Graph_Explainability.ipynb`
8. `Task_08_-_Graph_Intelligence_Dashboard.ipynb`
9. `Task_09_-_Graph_Transformer.ipynb` (bonus)
10. `Task_10_-_Advanced_GNN_Architecture.ipynb` (bonus)
11. `Task_11_-_Knowledge_Graph_Extension.ipynb` (bonus)
12. `Task_12_-_Self_Supervised_Graph_Learning.ipynb` (bonus)

Steps:

1. Open each notebook in Google Colab, in the order above (either upload it from your cloned folder, or open it directly from GitHub via File → Open notebook → GitHub).
2. Turn on a GPU runtime if available.
3. Mount Google Drive when asked — this is where shared outputs (dataset, results, models, figures) are saved and read between notebooks.
4. Run each notebook fully, top to bottom, before moving to the next one.
5. Check that the dataset, results, models, figures, and dashboard folders are created after Task 01–08.

The OGBN-Arxiv dataset downloads automatically the first time it's needed.

### Run the dashboard

Run Tasks 01–08 first — the dashboard needs files created by Tasks 02, 05, 06, and 07. Then run:

```bash
streamlit run dashboard/app.py
```

The dashboard has four pages:
- **Graph Statistics** – Dataset size, density, degree distributions, sample subgraph
- **Model Performance** – Validation/test metrics and training curves
- **Node Classification** – Predictions and confidence for selected papers
- **Embeddings** – PCA visualization of learned representations

## Project Structure

```text
OGBN_Arxiv_Project/
├── README.md
├── notebooks/
│   ├── .gitkeep
│   ├── Task_01_-_Tensor_Fundamentals.ipynb
│   ├── Task_02_-_Graph_Representation.ipynb
│   ├── Task_03_-_Graph_Data_Preparation.ipynb
│   ├── Task_04_-_Graph_Neural_Network.ipynb
│   ├── Task_05_-_Model_Training.ipynb
│   ├── Task_06_-_Model_Evaluation.ipynb
│   ├── Task_07_-_Graph_Explainability.ipynb
│   ├── Task_08_-_Graph_Intelligence_Dashboard.ipynb
│   ├── Task_09_-_Graph_Transformer.ipynb
│   ├── Task_10_-_Advanced_GNN_Architecture.ipynb
│   ├── Task_11_-_Knowledge_Graph_Extension.ipynb
│   └── Task_12_-_Self_Supervised_Graph_Learning.ipynb
├── dashboard/
│   └── app.py
├── data/
├── models/
│   ├── best_gcn.pt
│   └── best_graphsage.pt
└── results/
    ├── task02_graph_analysis/
    ├── task03_data_preparation/
    ├── task04_gnn_development/
    ├── task05_training/
    ├── task06_evaluation/
    ├── task07_explainability/
    └── task08_dashboard/
```

## Limitations

- The 40 subject categories are imbalanced, which affects macro metrics.
- Some related research areas have overlapping features and citation patterns, making them harder to tell apart.
- PCA compresses high-dimensional embeddings down to 2 dimensions, so some detail is lost.
- Neighbourhood agreement shows correlation, not proof of *why* the model made a decision.
- The Graph Transformer, GATv2, and self-supervised experiments run on a smaller sample, not the full graph.
- The knowledge graph extension is a small demonstration, not a full conversion of the dataset.

