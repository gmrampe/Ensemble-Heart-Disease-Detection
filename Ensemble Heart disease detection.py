#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import missingno as ms
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import io, base64


# read data set
def dataset_read():
    data = pd.read_csv("heart2.csv")
    y = data['target']
    data = data.drop('target', axis=1)
    x = data
    return x, y


# train_test method
def train_test(x, y, state):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=state)
    return x_train, x_test, y_train, y_test


# adaboost algorithm code
def ada_boost_classifier(x_train, x_test, y_train, y_test):
    model = AdaBoostClassifier()
    ada = model
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    traina = model.score(x_train, y_train)
    testa = model.score(x_test, y_test)
    print("Training Accuracy :", model.score(x_train, y_train))
    print("Testing Accuracy :", model.score(x_test, y_test))
    cm = confusion_matrix(y_test, prediction)
    cr = classification_report(y_test, prediction)
    print(cr, cm)
    return traina, testa, cm, cr


# naive bayes code:
def naive_baye(x_train, x_test, y_train, y_test):
    model = GaussianNB()
    nav = model
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    traina = model.score(x_train, y_train)
    testa = model.score(x_test, y_test)
    print("Training Accuracy :", model.score(x_train, y_train))
    print("Testing Accuracy :", model.score(x_test, y_test))
    cm = confusion_matrix(y_test, prediction)
    cr = classification_report(y_test, prediction)
    print(cr, cm)
    return traina, testa, cm, cr


# logistic regression code:
def logistic_regression(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
    lcr = model
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    traina = model.score(x_train, y_train)
    testa = model.score(x_test, y_test)
    print("Training Accuracy :", model.score(x_train, y_train))
    print("Testing Accuracy :", model.score(x_test, y_test))
    cm = confusion_matrix(y_test, prediction)
    cr = classification_report(y_test, prediction)
    print(cr, cm)
    return traina, testa, cm, cr


app = Flask(__name__)


@app.route("/")
def h():
    return render_template('index.html')


@app.route("/submit", methods=['POST'])
def submit():
    if request.method == "POST":
        value = request.form["value"]
        n = list(value)
        n.remove('v')
        n.remove('a')
        n.remove('l')
        s = [str(i) for i in n]
        print(n)
        num = int("".join(s))

        print(num)
        x, y = dataset_read()
        x_train, x_test, y_train, y_test = train_test(x, y, num)
        traina1, testa1, cm1, cr1 = ada_boost_classifier(x_train, x_test, y_train, y_test)
        traina2, testa2, cm2, cr2 = naive_baye(x_train, x_test, y_train, y_test)
        traina3, testa3, cm3, cr3 = logistic_regression(x_train, x_test, y_train, y_test)
        print("*" * 50)
        print("ramdom_state")
        print(num)
        print("final output is:")
        print("               " * 10)
    print('ada_boost_classifier')
    print(traina1, testa1, cm1, cr1)
    print("               " * 10)
    print('naive_baye')
    print(traina2, testa2, cm2, cr2)
    print("               " * 10)
    print('logistic_regression')
    print(traina3, testa3, cm3, cr3)

    dummy1 = "Train Accuracy is     :"
    dummy2 = "Test Accuracy is      :"
    dummy3 = "Confusion Matrix is   :"
    dummy4 = "classification Report :"
    data = [{
        'traina': dummy1 + str(traina),
        'testa': dummy2 + str(testa),
        'cm': dummy3 + str(cm),
        'cr': dummy4 + str(cr),

    },
        {
            'traina': dummy1 + str(traina1),
            'testa': dummy2 + str(testa1),
            'cm': dummy3 + str(cm1),
            'cr': dummy4 + str(cr1),

        },
        {
            'traina': dummy1 + str(traina2),
            'testa': dummy2 + str(testa2),
            'cm': dummy3 + str(cm2),
            'cr': dummy4 + str(cr2),

        },
        {
            'traina': dummy1 + str(traina3),
            'testa': dummy2 + str(testa3),
            'cm': dummy3 + str(cm3),
            'cr': dummy4 + str(cr3),

        },

    ]

    # Roc curve code:


classifiers = [AdaBoostClassifier(), GaussianNB(), LogisticRegression()]
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
x, y = dataset_read()
x_train, x_test, y_train, y_test = train_test(x, y, num)
for cls in classifiers:
    model = cls.fit(x_train, y_train)
    yproba = model.predict_proba(x_test)[::, 1]

    fpr, tpr, _ = roc_curve(y_test, yproba)
    auc = roc_auc_score(y_test, yproba)

    result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8, 6))
a = ['GaussianNB', 'AdaBoostClassifier', 'LogisticRegression']

j = 0

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(a[j], result_table.loc[i]['auc']))
    j = j + 1

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')

img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)

plot_url = base64.b64encode(img.getvalue()).decode()
plt.clf()

return render_template('index.html', data=data, plot_url=plot_url)
else:
return render_template('index.html')


@app.route("/fig", methods=['POST'])
def draw():
    value = request.form["plot_url"]
    return '<img src="data:image/png;base64,{}">'.format(value)


if __name__ == '__main__':
    app.run(debug=True)

