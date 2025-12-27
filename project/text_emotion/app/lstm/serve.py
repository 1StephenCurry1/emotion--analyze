"""Reload and serve a saved model"""
import json
import os

import jieba
from pathlib import Path
from functools import partial

# 尝试导入 TensorFlow，兼容不同版本
try:
    import tensorflow as tf
    # TensorFlow 2.x
    if hasattr(tf, 'saved_model'):
        TF_VERSION = 2
    else:
        TF_VERSION = 1
except ImportError:
    tf = None
    TF_VERSION = 0

LINE = '''酒店设施不是新的，服务态度很不好'''

# 全局模型缓存
_model = None
_signature = None


def load_model():
    """加载模型（懒加载，只加载一次）"""
    global _model, _signature
    if _model is not None:
        return _model, _signature
    
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model')
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    
    if TF_VERSION == 2:
        _model = tf.saved_model.load(latest)
        _signature = _model.signatures['serving_default']
    elif TF_VERSION == 1:
        from tensorflow.contrib import predictor
        _model = predictor.from_saved_model(latest)
        _signature = None
    
    return _model, _signature


def predict_tf2(line):
    """TensorFlow 2.x 预测"""
    model, signature = load_model()
    sentence = ' '.join(jieba.cut(line.strip(), cut_all=False, HMM=True))
    words = [w.encode() for w in sentence.strip().split()]
    nwords = len(words)
    
    # 构建输入
    words_tensor = tf.constant([words])
    nwords_tensor = tf.constant([nwords], dtype=tf.int32)
    
    # 预测
    result = signature(words=words_tensor, nwords=nwords_tensor)
    
    # 获取标签
    if 'labels' in result:
        label = result['labels'].numpy()[0].decode()
    else:
        # 尝试其他可能的输出键
        for key in result:
            if 'label' in key.lower():
                label = result[key].numpy()[0].decode()
                break
        else:
            label = list(result.values())[0].numpy()[0].decode()
    
    return label


def predict_tf1(line):
    """TensorFlow 1.x 预测"""
    model, _ = load_model()
    sentence = ' '.join(jieba.cut(line.strip(), cut_all=False, HMM=True))
    words = [w.encode() for w in sentence.strip().split()]
    nwords = len(words)
    predictions = model({'words': [words], 'nwords': [nwords]})
    return predictions['labels'].tolist()[0].decode()


def predict_mock(line):
    """模拟预测（当 TensorFlow 不可用时）"""
    # 简单的基于关键词的情感分析
    positive_words = ['开心', '高兴', '快乐', '喜欢', '爱', '好', '棒', '赞', '美', '优秀', '满意', '舒服', '方便', '干净']
    negative_words = ['差', '坏', '糟', '烂', '讨厌', '恨', '难受', '不好', '失望', '生气', '难过', '脏', '慢', '贵']
    
    pos_count = sum(1 for word in positive_words if word in line)
    neg_count = sum(1 for word in negative_words if word in line)
    
    if neg_count > pos_count:
        return 'NEG'
    else:
        return 'POS'


def predict_main(line):
    """主预测函数"""
    print(f"预测文本: {line}")
    
    try:
        if TF_VERSION == 2:
            result = predict_tf2(line)
        elif TF_VERSION == 1:
            result = predict_tf1(line)
        else:
            # TensorFlow 不可用，使用模拟预测
            print("警告: TensorFlow 未安装，使用简单关键词分析")
            result = predict_mock(line)
    except Exception as e:
        print(f"模型预测失败: {e}，使用简单关键词分析")
        result = predict_mock(line)
    
    print(f"预测结果: {result}")
    return result
