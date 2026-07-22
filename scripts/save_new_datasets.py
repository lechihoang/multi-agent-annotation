import pandas as pd

train_df = pd.read_csv('../data/ViCTSD_train.csv')
valid_df = pd.read_csv('../data/ViCTSD_valid.csv')
annotated_df = pd.read_csv('../data/ViCTSD_annotated.csv')

annotated_dict = dict(zip(annotated_df['text'], annotated_df['final_label']))

def replace_labels(df):
    new_df = df.copy()
    new_labels = []
    for text in new_df['Comment']:
        new_labels.append(annotated_dict.get(text, -1))
    new_df['Constructiveness'] = new_labels
    return new_df

train_new_df = replace_labels(train_df)
valid_new_df = replace_labels(valid_df)

train_new_df.to_csv('../data/ViCTSD_train_reannotated.csv', index=False)
valid_new_df.to_csv('../data/ViCTSD_valid_reannotated.csv', index=False)
print("Saved new datasets to data/ViCTSD_train_reannotated.csv and data/ViCTSD_valid_reannotated.csv")
