import pytest
import os
import pickle
import itertools

from sklearn import ensemble

def get_ada_randomforest_models():
    params = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [2, 10, 50, 100],
        'min_samples_split': [2, 5, 10, 50],
        'min_samples_leaf': [1, 2, 4, 50],
        'bootstrap': [True, False],
        'random_state': [0],

        'ada_n_estimators': [10, 50, 100, 200],
        'ada_learning_rate': [0.01, 0.1, 0.5, 1.0],
    }

    for param_values in itertools.product(*params.values()):
        params = dict(zip(params.keys(), param_values))
        model  = ensemble.AdaBoostClassifier(
            base_estimator=ensemble.RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                bootstrap=params['bootstrap'],
                random_state=params['random_state']),
            n_estimators=params['ada_n_estimators'],
            learning_rate=params['ada_learning_rate'],
            random_state=params['random_state'])

        yield {'name': 'ada_randomforest', 'model': model, 'params': params}

def get_randomforest_models():
    params = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [2, 10, 50, 100],
        'min_samples_split': [2, 5, 10, 50],
        'min_samples_leaf': [1, 2, 4, 50],
        'bootstrap': [True, False],
        'random_state': [0],
    }

    for param_values in itertools.product(*params.values()):
        params = dict(zip(params.keys(), param_values))
        model  = ensemble.RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            bootstrap=params['bootstrap'],
            random_state=params['random_state'])

        yield {'name': 'randomforest', 'model': model, 'params': params}

def get_gradientboosting_models():
    params = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'max_depth': [2, 10, 50, 100],
        'min_samples_split': [2, 5, 10, 50],
        'min_samples_leaf': [1, 2, 4, 50],
        'random_state': [0],
    }

    for param_values in itertools.product(*params.values()):
        params = dict(zip(params.keys(), param_values))
        model  = ensemble.GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state'])

        yield {'name': 'gradientboosting', 'model': model, 'params': params}

def get_ada_gradientboosting_models():
    params = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'max_depth': [2, 10, 50, 100],
        'min_samples_split': [2, 5, 10, 50],
        'min_samples_leaf': [1, 2, 4, 50],
        'random_state': [0],

        'ada_n_estimators': [10, 50, 100, 200],
        'ada_learning_rate': [0.01, 0.1, 0.5, 1.0],
    }

    for param_values in itertools.product(*params.values()):
        params = dict(zip(params.keys(), param_values))
        model  = ensemble.AdaBoostClassifier(
            base_estimator=ensemble.GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=params['random_state']),
            n_estimators=params['ada_n_estimators'],
            learning_rate=params['ada_learning_rate'],
            random_state=params['random_state'])

        yield {'name': 'ada_gradientboosting', 'model': model, 'params': params}

def get_voting_classifier_models():
    combinations = itertools.combinations()

@pytest.fixture(scope='module')
def output():
    result = os.path.join('output', 'models')
    if not os.path.exists(result):
        os.makedirs(result)
    return result

@pytest.mark.parametrize('model', get_ada_randomforest_models())
def test_ada_randomforest_models(request, output, model):
    pickle.dump(model, open(os.path.join(output, '{}.pkl'.format(request.node.name)), 'wb'))

@pytest.mark.parametrize('model', get_randomforest_models())
def test_randomforest_models(request, output, model):
    pickle.dump(model, open(os.path.join(output, '{}.pkl'.format(request.node.name)), 'wb'))

@pytest.mark.parametrize('model', get_gradientboosting_models())
def test_gradientboosting_models(request, output, model):
    pickle.dump(model, open(os.path.join(output, '{}.pkl'.format(request.node.name)), 'wb'))

@pytest.mark.parametrize('model', get_ada_gradientboosting_models())
def test_ada_gradientboosting_models(request, output, model):
    pickle.dump(model, open(os.path.join(output, '{}.pkl'.format(request.node.name)), 'wb'))
