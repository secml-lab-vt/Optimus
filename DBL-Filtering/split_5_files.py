"""Merge Advanced_Detect score CSVs back into per-trial Processed_files by uid."""

import os

import numpy as np
from datasets import Dataset, load_dataset

from common import PROCESSED_TO_ADV, dbl_path, ensure_file_parent


def merge_left_in_order(x, y, on=None):
    x = x.copy()
    x['Order'] = np.arange(len(x))
    z = x.merge(y, how='left', on=on)
    return z.sort_values('Order').drop('Order', axis=1)


def adv_keyword_for_processed(base_file):
    for key, adv in PROCESSED_TO_ADV:
        if key in base_file:
            return adv
    raise ValueError(f'No adv attack mapping for processed file: {base_file}')


def find_reference_score(scores_dir, adv_keyword):
    for name in os.listdir(scores_dir):
        if adv_keyword in name and name.endswith('.csv'):
            return os.path.join(scores_dir, name)
    raise FileNotFoundError(f'No score CSV matching {adv_keyword} in {scores_dir}')


def main():
    scores_dir = dbl_path('Advanced_Detect', 'Fewshot-no', '1')
    processed_dir = dbl_path('Processed_files')
    destination_path = dbl_path('Filtered_files')
    os.makedirs(destination_path, exist_ok=True)

    for file in sorted(os.listdir(processed_dir)):
        if not file.endswith('.csv'):
            continue

        adv_keyword = adv_keyword_for_processed(file)
        reference_file = find_reference_score(scores_dir, adv_keyword)

        reference_data = load_dataset('csv', data_files=reference_file, split='train')
        reference_data = reference_data.map(lambda x: {'uid': str(x['uid'])})

        original_data = load_dataset('csv', data_files=os.path.join(processed_dir, file), split='train')

        df1 = reference_data.to_pandas()
        df2 = original_data.to_pandas()
        df1['uid'] = df1['uid'].astype(str).str.replace('tensor(', '', regex=False).str.replace(')', '', regex=False)
        df2['uid'] = df2['uid'].astype(str)

        merged_df = merge_left_in_order(df2, df1, on='uid')
        merged = Dataset.from_pandas(merged_df)
        merged = merged.remove_columns(['contexts', 'responses', 'flags_y'])
        merged = merged.rename_column('flags_x', 'flags')

        out_path = os.path.join(destination_path, file)
        ensure_file_parent(out_path)
        merged.to_csv(out_path, index=False)
        print(f'Wrote {out_path} ({len(merged)} rows)')


if __name__ == '__main__':
    main()
