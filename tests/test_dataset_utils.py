import pandas as pd
from dataset_utils import build_text_label_dfs


def make_sample_text_df():
    texts = [
        {'text': 'Plants showing drought and dry soil', 'labels': [[0]], 'dataset': 'AG'},
        {'text': 'Yellow leaves due to nitrogen lack', 'labels': [[1]], 'dataset': 'DB'},
        {'text': 'Aphid infestation on leaves', 'labels': [[2]], 'dataset': 'Yahoo'},
        {'text': 'Rust and blight disease observed', 'labels': [[3]], 'dataset': 'DB'},
        {'text': 'Heat scorching on edges', 'labels': [[4]], 'dataset': 'Ag'},
    ]
    return pd.DataFrame(texts)


def test_build_text_label_dfs_minimum_counts():
    df = make_sample_text_df()
    keywords = {
        'water_stress': ['drought', 'dry'],
        'nutrient_def': ['nitrogen', 'fertil'],
        'pest_risk': ['aphid', 'pest'],
        'disease_risk': ['rust', 'blight', 'disease'],
        'heat_stress': ['heat', 'scorch']
    }
    results = build_text_label_dfs(df, keywords, min_per_label=3)
    # Each label should have at least 3 rows after synthesis if needed
    for lbl in keywords.keys():
        assert lbl in results
        assert len(results[lbl]) >= 3
