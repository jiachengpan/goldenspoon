import sklearn
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.mixture import *
from sklearn.svm import *
from sklearn.neural_network import *
from sklearn.neighbors import *
from sklearn.metrics import *
import xgboost as xgb

def get_model(model_name):
    if model_name == 'baseline':
        estimators=[
            ('rf1', RandomForestClassifier(max_depth=None, n_estimators=100, random_state=0)),
            #('rf2', RandomForestClassifier(max_depth=None, n_estimators=300, random_state=0)),
            ('ada', AdaBoostClassifier(random_state=0)),
            ('svm', SVC(kernel='rbf', random_state=0, probability=True)),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='mlogloss', random_state=0)),
            ('bst', GradientBoostingClassifier(random_state=0)),
            #('ada_lg', AdaBoostClassifier(base_estimator=LogisticRegression(), random_state=0)),
            #('lr',  LogisticRegression(multi_class='multinomial', random_state=0)),
            #('gnb',  GaussianNB()),
            #('knn',  KNeighborsClassifier(n_neighbors=3)),
            #('mlp',  MLPClassifier(random_state=0)),
        ]
        return VotingClassifier(estimators, voting='soft')
    else:
        assert False, f'unknown model name: {model_name}'