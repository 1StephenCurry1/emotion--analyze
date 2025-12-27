"""
中文情感分析服务

优先使用 SnowNLP（对中文效果稳定），备选 Transformers 和关键词匹配
"""
import os

# 全局模型缓存
_classifier = None
_model_type = None  # 'snownlp' 或 'transformers' 或 'mock'


def _load_snownlp_model():
    """加载 SnowNLP 模型（中文效果最稳定）"""
    global _classifier, _model_type
    
    try:
        from snownlp import SnowNLP
        _classifier = SnowNLP
        _model_type = 'snownlp'
        print("✓ 使用 SnowNLP 进行情感分析（中文效果稳定）")
        return True
    except ImportError:
        print("snownlp 库未安装，尝试其他方案...")
        return False


def _load_transformers_model():
    """加载 HuggingFace Transformers 模型（备选）"""
    global _classifier, _model_type
    
    try:
        from transformers import pipeline
        
        print("正在加载预训练情感分析模型...")
        
        # 真正的情感分析模型
        sentiment_models = [
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        ]
        
        for model_name in sentiment_models:
            try:
                print(f"尝试加载: {model_name}")
                _classifier = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                _model_type = 'transformers'
                print(f"✓ 成功加载模型: {model_name}")
                return True
            except Exception as e:
                print(f"加载 {model_name} 失败: {e}")
                continue
        
        return False
                
    except ImportError:
        print("transformers 库未安装")
        return False
    except Exception as e:
        print(f"加载 Transformers 模型失败: {e}")
        return False


def _init_model():
    """初始化模型（懒加载）"""
    global _classifier, _model_type
    
    if _classifier is not None:
        return
    
    # 优先使用 SnowNLP（中文效果稳定）
    if _load_snownlp_model():
        return
    
    # 备选 Transformers
    if _load_transformers_model():
        return
    
    # 都失败了，使用关键词匹配
    _model_type = 'mock'
    print("⚠ 使用关键词匹配（建议安装 snownlp: pip install snownlp）")


def _predict_snownlp(text):
    """使用 SnowNLP 预测（中文效果好）"""
    from snownlp import SnowNLP
    s = SnowNLP(text)
    score = s.sentiments  # 0-1 之间，越大越积极
    
    if score >= 0.5:
        return 'POS', score
    else:
        return 'NEG', 1 - score


def _predict_transformers(text):
    """使用 Transformers 模型预测"""
    result = _classifier(text)[0]
    label = result['label'].upper()
    score = result['score']
    
    # 统一标签格式
    if label in ['POSITIVE', 'POS', 'LABEL_1', 'LABEL_2']:
        return 'POS', score
    elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
        return 'NEG', score
    elif label in ['NEUTRAL', 'NEU']:
        return 'POS' if score > 0.5 else 'NEG', score * 0.6
    else:
        # 未知标签（可能是错误的模型），回退到 SnowNLP
        print(f"⚠ 未知标签 '{label}'，回退到 SnowNLP")
        return _predict_snownlp(text)


def _predict_mock(text):
    """
    增强版关键词情感分析
    
    改进点：
    1. 更丰富的情感词典
    2. 否定词处理
    3. 程度副词加权
    4. 双重否定处理
    """
    # 情感词典
    positive_words = {
        '开心', '高兴', '快乐', '喜欢', '爱', '好', '棒', '赞', '美', '优秀',
        '满意', '舒服', '方便', '干净', '推荐', '值得', '不错', '完美', '精彩',
        '感谢', '期待', '惊喜', '温馨', '贴心', '专业', '热情', '周到', '整洁',
        '宽敞', '安静', '便宜', '实惠', '划算', '超值', '给力', '厉害', '牛',
        '可以', '行', '成功', '顺利', '方便', '快速', '及时', '准时', '新鲜',
        '好吃', '美味', '香', '甜', '漂亮', '帅', '美丽', '可爱', '有趣', '好玩'
    }
    
    negative_words = {
        '差', '坏', '糟', '烂', '讨厌', '恨', '难受', '失望', '生气', '难过',
        '脏', '慢', '贵', '垃圾', '无语', '后悔', '坑', '骗', '假', '差评',
        '投诉', '退款', '吵', '臭', '破', '旧', '小', '挤', '冷', '热',
        '难吃', '苦', '酸', '咸', '淡', '腻', '硬', '软', '生', '焦',
        '丑', '土', '俗', '无聊', '累', '困', '烦', '急', '怕', '担心',
        '问题', '故障', '错误', '失败', '取消', '延迟', '缺货', '售罄'
    }
    
    # 否定词
    negation_words = {'不', '没', '无', '非', '别', '莫', '未', '毫无', '并非', '从未', '绝非'}
    
    # 程度副词（加权）
    degree_words = {
        '很': 1.5, '非常': 2.0, '特别': 2.0, '极其': 2.5, '太': 1.8,
        '超': 1.8, '真': 1.5, '好': 1.3, '挺': 1.2, '比较': 1.1,
        '有点': 0.8, '稍微': 0.7, '略': 0.6
    }
    
    # 分词（简单按字符和常见词切分）
    import jieba
    words = list(jieba.cut(text))
    
    pos_score = 0
    neg_score = 0
    
    # 遍历分析
    i = 0
    while i < len(words):
        word = words[i]
        
        # 检查程度副词
        degree = 1.0
        if word in degree_words:
            degree = degree_words[word]
            i += 1
            if i >= len(words):
                break
            word = words[i]
        
        # 检查否定词
        negated = False
        if word in negation_words:
            negated = True
            i += 1
            if i >= len(words):
                break
            # 检查双重否定
            if words[i] in negation_words:
                negated = False
                i += 1
                if i >= len(words):
                    break
            word = words[i]
        
        # 计算情感得分
        if word in positive_words:
            if negated:
                neg_score += degree
            else:
                pos_score += degree
        elif word in negative_words:
            if negated:
                pos_score += degree
            else:
                neg_score += degree
        
        i += 1
    
    # 判断结果
    total = pos_score + neg_score
    if total == 0:
        return 'POS', 0.5  # 中性默认为积极
    
    if pos_score > neg_score:
        return 'POS', pos_score / total
    else:
        return 'NEG', neg_score / total


def predict_main(line):
    """
    主预测函数
    
    Args:
        line: 待分析的文本
        
    Returns:
        str: 'POS'（积极）或 'NEG'（消极）
    """
    print(f"预测文本: {line}")
    
    # 初始化模型
    _init_model()
    
    try:
        if _model_type == 'snownlp':
            result, confidence = _predict_snownlp(line)
        elif _model_type == 'transformers':
            result, confidence = _predict_transformers(line)
        else:
            result, confidence = _predict_mock(line)
        
        print(f"预测结果: {result} (置信度: {confidence:.2%})")
        print(f"使用模型: {_model_type}")
        return result
        
    except Exception as e:
        print(f"预测失败: {e}，使用关键词分析")
        result, _ = _predict_mock(line)
        return result


def get_detailed_result(line):
    """
    获取详细的预测结果（包含置信度）
    
    Args:
        line: 待分析的文本
        
    Returns:
        dict: {'label': 'POS'/'NEG', 'confidence': 0.0-1.0, 'model': 'xxx'}
    """
    _init_model()
    
    try:
        if _model_type == 'snownlp':
            label, confidence = _predict_snownlp(line)
        elif _model_type == 'transformers':
            label, confidence = _predict_transformers(line)
        else:
            label, confidence = _predict_mock(line)
        
        return {
            'label': label,
            'confidence': confidence,
            'model': _model_type,
            'text': line
        }
    except Exception as e:
        label, confidence = _predict_mock(line)
        return {
            'label': label,
            'confidence': confidence,
            'model': 'mock',
            'error': str(e),
            'text': line
        }


# 测试代码
if __name__ == '__main__':
    test_texts = [
        "这个酒店非常好，服务态度很棒！",
        "酒店设施不是新的，服务态度很不好",
        "房间很干净，位置也方便",
        "太贵了，不值这个价",
        "一般般吧，没什么特别的",
        "强烈推荐，下次还会来",
        "差评，再也不来了",
        "不是很满意，但也不算太差",
        "今天心情真好",
        "这代码写得真烂"
    ]
    
    print("=" * 60)
    print("情感分析测试")
    print("=" * 60)
    
    for text in test_texts:
        result = get_detailed_result(text)
        emoji = "😊" if result['label'] == 'POS' else "😞"
        print(f"\n{emoji} {text}")
        print(f"   结果: {result['label']} | 置信度: {result['confidence']:.2%} | 模型: {result['model']}")
