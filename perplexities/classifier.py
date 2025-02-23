import pandas as pd
from ipykernel.kernelapp import kernel_aliases
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


def perp_translate(row):
    if 'shuffle_control' in row['perturb']:
        return 1
    else:
        return 0

perplexity_result = pd.read_csv('perplexity_results.csv')

perplexity_result['possible'] = perplexity_result.apply(perp_translate, axis=1)
perplexity_result.drop(perplexity_result[perplexity_result.perturb.str.contains('adj')].index, inplace=True)

checkpoint_columns = [col for col in perplexity_result.columns if 'checkpoint' in col]
aggregated = perplexity_result.groupby(['perturb', 'lang'])[checkpoint_columns].mean().reset_index()
aggregated['possible'] = perplexity_result.groupby(['perturb', 'lang'])['possible'].mean().reset_index()['possible']
aggregated['seed'] = "average"
perplexity_result = aggregated

perplexity_result.drop(['lang', 'seed', 'perturb'], axis=1, inplace=True)
# perplexity_result = (perplexity_result - perplexity_result.min()) / (perplexity_result.max() - perplexity_result.min())

checkpoints = [f'checkpoint{str(n)}' for n in range(0, 1201,100)]

impossible = perplexity_result[perplexity_result.possible==0]
possible = perplexity_result.drop(perplexity_result[perplexity_result.possible==0].index)

parameters = {'kernel': ('linear', 'poly'),  'C':range(0,11), 'gamma':('scale', 'auto'),}

X = perplexity_result[checkpoints]
y = perplexity_result['possible']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

clf= svm.SVC(kernel='linear', C=1, gamma='auto')
# clf = GridSearchCV(clf, parameters, verbose=True)
clf.fit(X_train, y_train)

print(sum(cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro'))/5)

y_pred = clf.predict(X_test)
print(pd.crosstab(y_test, y_pred))
print(classification_report(y_test, y_pred))
