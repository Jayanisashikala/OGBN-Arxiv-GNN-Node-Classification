
import json
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK02 = PROJECT_ROOT / 'results' / 'task02_graph_analysis'
TASK05 = PROJECT_ROOT / 'results' / 'task05_training'
TASK06 = PROJECT_ROOT / 'results' / 'task06_evaluation'
TASK07 = PROJECT_ROOT / 'results' / 'task07_explainability'

st.set_page_config(page_title='OGBN-Arxiv Dashboard', layout='wide')
st.title('OGBN-Arxiv Graph Intelligence Dashboard')
page = st.sidebar.radio('Page', ['Graph statistics', 'Model performance', 'Node classification', 'Embeddings'])

if page == 'Graph statistics':
    st.header('Graph statistics')
    stats = json.loads((TASK02 / 'graph_statistics.json').read_text())
    a,b,c,d = st.columns(4)
    a.metric('Papers', f"{stats['nodes']:,}"); b.metric('Citations', f"{stats['edges']:,}")
    c.metric('Features', stats['features']); d.metric('Classes', stats['classes'])
    st.metric('Graph density', f"{stats['density']:.10f}")
    st.image(str(TASK02 / 'degree_distribution.png'))
    st.image(str(TASK02 / 'sample_subgraph.png'))
elif page == 'Model performance':
    st.header('Validation and test metrics')
    results = pd.read_csv(TASK06 / 'model_evaluation.csv')
    st.dataframe(results, use_container_width=True)
    st.bar_chart(results[results.split == 'Test'].set_index('model')[['accuracy','precision_macro','recall_macro','f1_macro']])
    st.image(str(TASK05 / 'best_model_accuracy_curves.png'))
elif page == 'Node classification':
    results = pd.read_csv(TASK06 / 'node_classification_results.csv')
    model = st.selectbox('Model', sorted(results.model.unique()))
    selected = results[results.model == model]
    node = st.selectbox('Node ID', selected.node_id.tolist())
    st.dataframe(selected[selected.node_id == node], use_container_width=True)
else:
    st.header('PCA node embeddings')
    st.image(str(TASK07 / 'pca_embeddings.png'))
    st.dataframe(pd.read_csv(TASK07 / 'pca_embeddings.csv').head(100), use_container_width=True)
