"""Reload and serve a saved model"""
import json
import os

import jieba
from pathlib import Path
from tensorflow.contrib import predictor
from functools import partial

LINE = '''酒店设施不是新的，服务态度很不好'''


def predict(pred_fn, line):
    sentence = ' '.join(jieba.cut(line.strip(), cut_all=False, HMM=True))
    words = [w.encode() for w in sentence.strip().split()]
    nwords = len(words)
    predictions = pred_fn({'words': [words], 'nwords': [nwords]})
    return predictions


def predict_main(line):
    print(line)
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'saved_model')
    print(Path(export_dir))
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    print(subdirs)
    latest = str(sorted(subdirs)[-1])
    predict_fn = partial(predict, predictor.from_saved_model(latest))

    result = predict_fn(line)['labels'].tolist()[0].decode()
    print(result)
    return result
