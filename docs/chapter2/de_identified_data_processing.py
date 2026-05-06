import re
from typing import List, Callable
from transformers import pipeline


# 1. 深度学习模型初始化
# 初始化命名实体识别（NER）流水线
ner_pipeline = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    aggregation_strategy="simple"  # 替换为当前支持的参数
)

def ner_mask(text: str) -> str:
    """
    利用深度学习模型进行语义级别的脱敏（人名与地名）
    """
    entities = ner_pipeline(text)
    spans = []
    
    # 提取模型识别出的实体及其位置
    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]

        # 映射实体类型到脱敏占位符
        if label == "PER":  # Person: 人名
            spans.append((start, end, "[NAME]"))
        elif label == "LOC":  # Location: 地名/地址
            spans.append((start, end, "[PLACE]"))

    # 排序逻辑：按起始位置升序；如果起始位置相同，按长度降序（优先处理长实体）
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # 解决冲突：去除重叠或包含关系的实体区间
    filtered_spans = []
    last_end = -1
    for start, end, tag in spans:
        if start >= last_end:  # 只有当当前实体起始位置在上一实体结束之后，才保留
            filtered_spans.append((start, end, tag))
            last_end = end

    # 根据过滤后的区间重建文本
    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start]) # 拼接非敏感部分
        result.append(tag)                  # 拼接占位符
        last_idx = end
    result.append(text[last_idx:])          # 拼接剩余文本

    return "".join(result)


# 2. 脱敏流水线架构设计
class DesensitizationPipeline:
    """
    脱敏任务管理器：允许按顺序添加多个处理步骤
    """
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        """添加处理环节（如正则替换、NER替换等）"""
        self.steps.append(func)

    def run(self, text: str) -> str:
        """按顺序执行所有脱敏步骤"""
        for step in self.steps:
            text = step(text)
        return text

# 3. 具体处理步骤实现
def normalize_text(text: str) -> str:
    """文本预处理：去除首尾空格"""
    return text.strip()

# 高确定性规则（强特征：手机号、邮箱）
def mask_phone(text: str) -> str:
    """正则匹配 11 位中国手机号"""
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

def mask_email(text: str) -> str:
    """正则匹配常见邮箱格式"""
    return re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)

# 中确定性规则（基于关键词上下文）
def mask_address(text: str) -> str:
    """通过“居住于”等关键词引导的地址匹配"""
    return re.sub(
        r'(居住于|现居住于|现居于|地址)([\u4e00-\u9fa5A-Za-z0-9]+)',
        r'\1[PLACE]',
        text
    )

# 低确定性规则（基于语法结构的简单兜底）
def mask_name(text: str) -> str:
    """
    兜底策略：匹配出现在句首或标点后的“某某某的”结构
    注：容易误伤，通常放在 NER 步骤之后作为补充
    """
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？]))([\u4e00-\u9fa5]{2,3})(的)',
        r'[NAME]\2',
        text
    )

def clean_punctuation(text: str) -> str:
    """后处理环节：可根据需求规范化标点符号"""
    return text


# 4. 构建与测试
def build_pipeline():
    """
    组装流水线：建议遵循“预处理 -> 高准确率正则 -> AI识别 -> 低准确率正则兜底”的顺序
    """
    p = DesensitizationPipeline()

    # 基础清理
    p.add_step(normalize_text)

    # 静态规则（手机、邮箱这类模式固定的最先处理，防止被NER误切）
    p.add_step(mask_phone)
    p.add_step(mask_email)

    # AI语义识别（主力：处理复杂的人名、地名）
    p.add_step(ner_mask)

    # 动态规则兜底（针对模型可能漏掉的特定话术）
    p.add_step(mask_address)
    p.add_step(mask_name)

    # 收尾处理
    p.add_step(clean_punctuation)

    return p

if __name__ == "__main__":
    # 测试用例：包含人名、邮箱、电话、地名及详细地址
    test_text = "小明的邮箱是test@gmail.com，电话是13312311111，现在居住于重庆两江新区的xxx小区。"
    
    ds_pipeline = build_pipeline()

    print("--- 脱敏系统测试 ---")
    print("处理前:", test_text)
    print("处理后:", ds_pipeline.run(test_text))
