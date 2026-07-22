import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
train_orig = pd.read_csv('data/ViCTSD_train.csv').fillna('')
train_new = pd.read_csv('data/ViCTSD_train_reannotated.csv').fillna('')
test_df = pd.read_csv('data/ViCTSD_test.csv').fillna('')

train_new = train_new[train_new['Constructiveness'] != -1]

# Vectorize
vec_orig = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_orig = vec_orig.fit_transform(train_orig['Comment'])
X_test_orig = vec_orig.transform(test_df['Comment'])
y_train_orig = train_orig['Constructiveness'].astype(int)

vec_new = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_new = vec_new.fit_transform(train_new['Comment'])
X_test_new = vec_new.transform(test_df['Comment'])
y_train_new = train_new['Constructiveness'].astype(int)

y_test = test_df['Constructiveness'].astype(int)

results = []

# Logistic Regression
lr_orig = LogisticRegression(max_iter=1000)
lr_orig.fit(X_train_orig, y_train_orig)
y_pred_lr_orig = lr_orig.predict(X_test_orig)
results.append({'Model': 'Logistic Regression', 'Dataset': 'Original Labels', 
                'Accuracy': accuracy_score(y_test, y_pred_lr_orig),
                'F1 Macro': f1_score(y_test, y_pred_lr_orig, average='macro')})

lr_new = LogisticRegression(max_iter=1000)
lr_new.fit(X_train_new, y_train_new)
y_pred_lr_new = lr_new.predict(X_test_new)
results.append({'Model': 'Logistic Regression', 'Dataset': 'AI Annotated Labels', 
                'Accuracy': accuracy_score(y_test, y_pred_lr_new),
                'F1 Macro': f1_score(y_test, y_pred_lr_new, average='macro')})

# XGBoost
xgb_orig = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_orig.fit(X_train_orig, y_train_orig)
y_pred_xgb_orig = xgb_orig.predict(X_test_orig)
results.append({'Model': 'XGBoost', 'Dataset': 'Original Labels', 
                'Accuracy': accuracy_score(y_test, y_pred_xgb_orig),
                'F1 Macro': f1_score(y_test, y_pred_xgb_orig, average='macro')})

xgb_new = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_new.fit(X_train_new, y_train_new)
y_pred_xgb_new = xgb_new.predict(X_test_new)
results.append({'Model': 'XGBoost', 'Dataset': 'AI Annotated Labels', 
                'Accuracy': accuracy_score(y_test, y_pred_xgb_new),
                'F1 Macro': f1_score(y_test, y_pred_xgb_new, average='macro')})

results_df = pd.DataFrame(results)

# Plotting
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Dataset', ax=axes[0], palette='viridis')
axes[0].set_title('Accuracy Comparison (Test Set = Original Humans)')
axes[0].set_ylim(0, 1.0)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

sns.barplot(data=results_df, x='Model', y='F1 Macro', hue='Dataset', ax=axes[1], palette='viridis')
axes[1].set_title('F1 Macro Comparison (Test Set = Original Humans)')
axes[1].set_ylim(0, 1.0)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300)
print("Chart saved to model_comparison_chart.png")
print("\n--- RESULTS SUMMARY ---")
print(results_df.to_string(index=False))

