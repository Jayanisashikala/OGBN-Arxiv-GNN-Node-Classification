# OGBN-Arxiv Graph Neural Network Node Classification

This project applies tensor operations, graph analytics, Graph Neural Networks (GNNs), explainability methods, and a Streamlit dashboard to the **OGBN-Arxiv citation network**. It was developed for the **CCS4354 - Tensors and Graphs** coursework at SLTC Research University.

The main objective is to predict the computer-science subject category of each research paper using its 128-dimensional feature vector and citation relationships.

## Dataset

The project uses the [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset from the Open Graph Benchmark.

| Property | Value |
|---|---:|
| Nodes | 169,343 papers |
| Directed edges | 1,166,243 citations |
| Node features | 128 |
| Target classes | 40 |
| Task | Multi-class node classification |
| Split | Official chronological OGB split |

Each node represents a scientific paper. A directed edge from paper A to paper B means that paper A cites paper B. The node features are numerical embeddings generated from paper titles and abstracts.

## Project Tasks

### Compulsory tasks

1. **Tensor Fundamentals** - Tensor creation, indexing, reshaping, matrix multiplication, broadcasting, aggregation, and GPU operations.
2. **Graph Representation and Analysis** - Edge-list representation, node-feature analysis, degree distributions, graph density, connected components, and sample-subgraph visualization.
3. **Graph Data Preparation** - Official train, validation, and test splits; missing-value checks; feature preprocessing; and bidirectional message-passing edges.
4. **GNN Development** - Three-layer GCN and GraphSAGE models with batch normalization, ReLU, dropout, and residual connections.
5. **Training and Optimization** - Cross-entropy with label smoothing, Adam, weight decay, learning-rate scheduling, gradient clipping, early stopping, and hyperparameter comparison.
6. **Model Evaluation** - Accuracy, macro precision, macro recall, macro F1, confusion matrices, and node-level predictions.
7. **Explainability and Embeddings** - PCA visualization and neighbourhood-influence analysis.
8. **Graph Intelligence Dashboard** - Streamlit pages for graph statistics, model performance, node classifications, and embeddings.

### Bonus tasks

9. **Graph Transformer** - A sampled attention-based graph classification experiment.
10. **GATv2** - An advanced attention-based GNN evaluated on the same sampled graph.
11. **Knowledge Graph Extension** - A small paper-subject knowledge graph with `cites` and `belongs_to` relationships.
12. **Self-Supervised Graph Learning** - Masked-feature reconstruction followed by linear evaluation of frozen graph embeddings.

## Data Preparation

The project uses the official chronological split:

| Split | Nodes | Percentage |
|---|---:|---:|
| Training | 90,941 | 53.70% |
| Validation | 29,799 | 17.60% |
| Test | 48,603 | 28.70% |

The split groups are complete and non-overlapping. Feature statistics are calculated from training nodes only to prevent data leakage. The original OGB feature embeddings are retained after preprocessing comparison. Reverse edges are added only for GNN message passing, increasing the message-passing edge count to 2,315,598.

## Compulsory Models

### Graph Convolutional Network

The GCN applies normalized graph convolution across connected papers. It contains two hidden graph layers and one output graph layer.

### GraphSAGE

GraphSAGE uses mean neighbourhood aggregation. It combines each paper's own representation with information received from connected papers.

Both models use:

- Three graph message-passing layers
- Two batch-normalization layers
- ReLU activation
- Dropout
- A residual connection
- A 40-class output layer

## Training Configuration

The executed experiment compared two configurations for each architecture:

| Configuration | Hidden dimension | Learning rate | Dropout | Maximum epochs | Patience |
|---|---:|---:|---:|---:|---:|
| 1 | 128 | 0.010 | 0.5 | 150 | 20 |
| 2 | 256 | 0.005 | 0.4 | 150 | 20 |

The training workflow uses:

- Cross-entropy loss with label smoothing of 0.05
- Adam optimizer
- Weight decay of `5e-4`
- `ReduceLROnPlateau` learning-rate scheduler
- Gradient clipping with a maximum norm of 2.0
- Validation-based checkpoint selection
- Early stopping

The selected configuration for both models used 256 hidden units, a learning rate of 0.005, and dropout of 0.4.

## Results

### Compulsory model evaluation

| Model | Split | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| GCN | Validation | 71.67% | 60.86% | 46.04% | 49.21% |
| GCN | Test | 70.72% | 57.55% | 44.23% | 47.38% |
| GraphSAGE | Validation | 72.57% | 60.46% | 49.63% | 52.00% |
| GraphSAGE | Test | **70.85%** | **59.99%** | **47.67%** | **49.66%** |

GraphSAGE produced the strongest overall validation and test results. Both models have lower macro metrics than accuracy because the 40 subject categories are imbalanced.

### Explainability examples

| Node | True class | Predicted class | Confidence | Correct | Neighbour agreement |
|---:|---:|---:|---:|---|---:|
| 451 | 24 | 24 | 69.92% | Yes | 81.82% |
| 346 | 24 | 10 | 43.54% | No | 0.00% |

PCA provides a global view of the learned node embeddings. Neighbourhood analysis provides a local view by comparing a prediction with the labels of connected papers.

### Bonus results

| Method | Evaluation scope | Validation accuracy | Test accuracy/output |
|---|---|---:|---:|
| Graph Transformer | 10,000-node sample | 50.60% | 49.30% |
| GATv2 | 10,000-node sample | 52.85% | 51.15% |
| Knowledge graph | 12 training papers | Not applicable | Triple table and graph visualization |
| Self-supervised encoder | 10,000-node sample | 16.90% | 15.40% |

The Graph Transformer and GATv2 results are not directly comparable with the compulsory models because the attention models use a sampled subgraph. During self-supervised pretraining, reconstruction loss decreased from 0.0580 to 0.0138.

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- Open Graph Benchmark (`ogb`)
- NumPy
- pandas
- SciPy
- scikit-learn
- NetworkX
- Matplotlib
- seaborn
- Streamlit
- Google Colab

## Installation

Install the main dependencies in Google Colab or a compatible Python environment:

```bash
pip install torch torch-geometric ogb streamlit pandas numpy scipy scikit-learn networkx matplotlib seaborn
```

GPU acceleration is recommended for full-graph GCN and GraphSAGE training.

## Running the Notebook

1. Open `CCS4354_Assignment.ipynb` in Google Colab.
2. Enable a GPU runtime if one is available.
3. Mount Google Drive when the notebook requests it.
4. Run the notebook from the first cell to the final cell in order.
5. Confirm that the dataset, results, models, figures, tables, and dashboard folders are created.

The notebook downloads OGBN-Arxiv automatically on its first execution.

## Running the Dashboard

Run the notebook first because the dashboard reads the files generated by Tasks 2, 5, 6, and 7. Then run:

```bash
streamlit run dashboard/app.py
```

The dashboard contains four pages:

- **Graph statistics** - Dataset size, density, degree distributions, and sample subgraph
- **Model performance** - Validation/test metrics and training curves
- **Node classification** - True class, predicted class, and confidence for selected nodes
- **Embeddings** - PCA visualization and saved embedding coordinates

## Expected Project Structure

```text
OGBN_Arxiv_Project/
├── CCS4354_Assignment.ipynb
├── README.md
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

## Main Saved Outputs

- Graph statistics and structural-analysis tables
- Degree-distribution and sample-subgraph figures
- Split and preprocessing summaries
- Best GCN and GraphSAGE model checkpoints
- Hyperparameter-comparison results
- Training histories and accuracy curves
- Validation and test metrics
- Confusion matrices
- Node-level classification results
- PCA embedding coordinates and figures
- Neighbourhood-influence table
- Streamlit dashboard application

## Limitations

- OGBN-Arxiv contains imbalanced subject classes.
- Related research areas may have overlapping feature and citation patterns.
- PCA reduces high-dimensional embeddings to only two components.
- Neighbourhood agreement provides descriptive evidence, not causal proof.
- The attention and self-supervised bonus experiments use a sampled graph.
- The small knowledge graph demonstrates the concept but does not convert the complete dataset into a heterogeneous graph.

## Group Members

| Student ID | Name |
|---|---|
| CIT-23-02-0162 | Jayani Sashikala |
| CIT-23-02-0130 | Sandani Senevithna |
| CIT-23-02-0073 | Ganguli Kaluarachchi |
| CIT-23-02-0358 | Dureksha Arangala |

## Academic Integrity

This project was developed for academic coursework. Anyone reusing the repository should cite the original dataset and comply with their institution's academic-integrity requirements.

## References

- Hu, W. et al. *Open Graph Benchmark: Datasets for Machine Learning*. NeurIPS, 2020.
- Kipf, T. N. and Welling, M. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR, 2017.
- Hamilton, W. L., Ying, R., and Leskovec, J. *Inductive Representation Learning on Large Graphs*. NeurIPS, 2017.
- PyTorch Geometric documentation: <https://pytorch-geometric.readthedocs.io/>
- OGB documentation: <https://ogb.stanford.edu/docs/>
