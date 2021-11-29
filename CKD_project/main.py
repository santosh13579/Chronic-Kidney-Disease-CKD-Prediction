import flask
import pickle
import numpy as np
from utility_func.preprocessing import *
from flask import Flask, render_template, request
import pandas as pd

features_dict = {'age': {'NaN': 55.0},
            'bp': {'NaN': 80.0},
            'rbc': {'NaN': 0, 'abnormal': 1, 'normal': 2},
            'pc': {'NaN': 0, 'abnormal': 1, 'normal': 2},
			'sg': {'NaN': 0, '1.005': 1, '1.010': 2, '1.015': 3, '1.020': 4, '1.025': 5},
            'al': {'NaN': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6},
            'su': {'NaN': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6},
            'bgr': {'NaN': 121.0},
            'bu': {'NaN': 42.0},
            'sc': {'NaN': 1.3},
			'pcc': {'NaN': 0, 'notpresent': 1, 'present': 2},
            'ba': {'NaN': 0, 'notpresent': 1, 'present': 2},
            'sod': {'NaN': 138.0},
            'pot': {'NaN': 4.4},
            'hemo': {'NaN': 12.649999999999999},
            'pcv': {'NaN': 40.0},
            'wc': {'NaN': 8000.0},
            'rc': {'NaN': 4.8},
            'htn': {'NaN': 0, 'no': 1, 'yes': 2},
            'dm': {'NaN': 0, 'no': 1, 'yes': 2},
            'cad': {'NaN': 0, 'no': 1, 'yes': 2},
            'appet': {'NaN': 0, 'poor': 1, 'good': 2},
            'pe': {'NaN': 0, 'no': 1, 'yes': 2},
            'ane': {'NaN': 0, 'no': 1, 'yes': 2},
            }

main = Flask(__name__)


@main.route('/')
def entry():
    return render_template('form.html', the_title='Chronic Kidney Disease', form_title='Input Features')

@main.route('/analyse', methods=['GET', 'POST'])
def proc():
    test_val = create_df()
    for parameters in test_val.columns:
        if request.form[parameters]:
            test_val[parameters] = request.form[parameters]
        else:
            test_val[parameters] = 'NaN'
    remove_missing(test_val, features_dict)
    test_val = test_val.astype(float)
    result = {}
    model  = load_model()
    pred = model.predict(test_val)
    param= pd.read_csv('./model/feature_important.csv')
    result['prediction'] = str(pred).strip('[]')
    if pred == 1:
        result['inference'] = 'Patient has potential Chronic Kidney Disease!!! Please consult your Doctor immideately.'
    else:
        result['inference'] = 'Patient does not has Chronic Kidney Disease. Enjoy your life with full of Hapiness'
    return render_template('result.html',risky=param, pred=result['inference'])

if __name__ == '__main__':
    main.run(host="localhost", port=8080, debug=True)

