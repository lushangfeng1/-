# -*- coding: utf-8 -*-
"""
使用训练好的分位数回归模型对原始预测结果.csv中的数据进行预测和风险标注
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("使用训练好的模型对原始预测结果数据进行风险标注")
print("=" * 60)

# ============================================================================
# 第一步：加载训练数据并训练模型
# ============================================================================
print("\n1. 加载训练数据并训练模型...")

# 加载训练数据
train_df = pd.read_csv('train_processed.csv')
print(f"训练数据形状: {train_df.shape}")

# 创建数分位指标（训练数据）
train_df['数分位指标'] = train_df['理赔差额'] / (train_df['实际赔付金额'] + 1)
print(f"训练数据数分位指标统计:")
print(train_df['数分位指标'].describe())

# 准备特征（排除目标变量）
feature_cols = [col for col in train_df.columns if col not in ['索赔金额', '实际赔付金额', '数分位指标', '虚报比例']]
print(f"\n回归特征数量: {len(feature_cols)}")

# 处理训练特征数据
X_train = train_df[feature_cols].copy()

# 对分类变量进行编码（训练数据）
categorical_cols = X_train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    label_encoders[col] = le  # 保存编码器以便后续使用

# 填充缺失值
X_train = X_train.fillna(X_train.median())

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)

y_train = train_df['数分位指标'].copy()

print(f"训练特征矩阵形状: {X_train_scaled.shape}")
print(f"训练目标变量形状: {y_train.shape}")

# 训练分位数回归模型
print("\n2. 训练分位数回归模型...")
quantiles = [0.85, 0.97]
quantile_models = {}

for q in quantiles:
    print(f"  训练 {q*100:.0f}% 分位数模型...")
    model = QuantileRegressor(quantile=q, alpha=0.1)
    model.fit(X_train_scaled, y_train)
    quantile_models[q] = model
    print(f"    ✓ 完成")

print("\n模型训练完成！")

# ============================================================================
# 第二步：加载原始预测数据
# ============================================================================
print("\n3. 加载原始预测数据...")
predict_df = pd.read_csv('原始预测结果.csv')  # 此处改为原始预测结果.csv
print(f"原始预测数据形状: {predict_df.shape}")
print(f"原始预测数据列: {list(predict_df.columns)}")

# 检查必要的列是否存在
required_cols = ['索赔金额', '预测赔付金额']
missing_cols = [col for col in required_cols if col not in predict_df.columns]
if missing_cols:
    raise ValueError(f"预测数据缺少必要的列: {missing_cols}")

# 计算理赔差额和数分位指标（使用预测赔付金额代替实际赔付金额）
print("\n4. 计算理赔差额和数分位指标...")
predict_df['理赔差额'] = predict_df['索赔金额'] - predict_df['预测赔付金额']
predict_df['数分位指标'] = predict_df['理赔差额'] / (predict_df['预测赔付金额'] + 1)

print(f"原始预测数据理赔差额统计:")
print(predict_df['理赔差额'].describe())
print(f"\n原始预测数据数分位指标统计:")
print(predict_df['数分位指标'].describe())

# ============================================================================
# 第三步：预处理原始预测数据特征
# ============================================================================
print("\n5. 预处理原始预测数据特征...")

# 确保预测数据包含所有需要的特征列
# 如果预测数据缺少某些训练时使用的列，用默认值填充
X_predict = predict_df[feature_cols].copy() if all(col in predict_df.columns for col in feature_cols) else pd.DataFrame()

if X_predict.empty:
    # 如果特征列不匹配，尝试从预测数据中选择可用的特征
    available_features = [col for col in feature_cols if col in predict_df.columns]
    missing_features = [col for col in feature_cols if col not in predict_df.columns]

    print(f"可用特征数: {len(available_features)}")
    print(f"缺失特征数: {len(missing_features)}")
    if missing_features:
        print(f"缺失的特征: {missing_features[:10]}...")  # 只显示前10个

    # 使用可用特征，缺失的用0填充
    X_predict = predict_df[available_features].copy()
    for col in missing_features:
        X_predict[col] = 0  # 用0填充缺失特征
    # 确保列顺序与训练数据一致
    X_predict = X_predict[feature_cols]

# 对分类变量进行编码（使用训练数据的编码器）
for col in categorical_cols:
    if col in X_predict.columns:
        # 使用训练数据的编码器，未知值编码为-1或最大编码值+1
        try:
            X_predict[col] = X_predict[col].astype(str).map(lambda x: label_encoders[col].transform([x])[0] 
                                                             if x in label_encoders[col].classes_ 
                                                             else -1)
        except:
            max_label = len(label_encoders[col].classes_)
            X_predict[col] = X_predict[col].astype(str).apply(
                lambda x: max_label if x not in label_encoders[col].classes_ 
                else list(label_encoders[col].classes_).index(x)
            )

# 填充缺失值（使用训练数据的median）
X_predict = X_predict.fillna(X_train.median())

# 标准化特征（使用训练数据的scaler）
X_predict_scaled = scaler.transform(X_predict)
X_predict_scaled = pd.DataFrame(X_predict_scaled, columns=feature_cols)

print(f"原始预测特征矩阵形状: {X_predict_scaled.shape}")

# ============================================================================
# 第四步：使用模型进行预测和风险标注
# ============================================================================
print("\n6. 使用模型对原始预测结果进行风险标注...")

# 使用训练好的模型进行预测
predictions = {}
for q, model in quantile_models.items():
    pred = model.predict(X_predict_scaled)
    predictions[q] = pred
    print(f"  {q*100:.0f}%分位数预测范围: [{pred.min():.4f}, {pred.max():.4f}]")

# 基于预测的分位数阈值进行风险标注
def assign_risk_label(row):
    """基于数分位指标和分位数预测进行风险标注"""
    actual_value = row['数分位指标']
    q85_pred = row['q85_pred']
    q97_pred = row['q97_pred']

    # 风险标注规则：
    # 小于85%分位数 -> 合理诉求
    # 85%-97%分位数 -> 诉求偏高
    # 大于97%分位数 -> 严重超额
    if actual_value <= q85_pred:
        return "合理诉求"
    elif actual_value <= q97_pred:
        return "诉求偏高"
    else:
        return "严重超额"

# 创建预测结果DataFrame
pred_df = pd.DataFrame({
    'q85_pred': predictions[0.85],
    'q97_pred': predictions[0.97],
    '数分位指标': predict_df['数分位指标']
})

# 应用风险标注
predict_df['风险标注'] = pred_df.apply(assign_risk_label, axis=1)

# ============================================================================
# 第五步：统计结果并保存
# ============================================================================
print("\n7. 风险标注结果统计...")
risk_counts = predict_df['风险标注'].value_counts()
risk_percentages = predict_df['风险标注'].value_counts(normalize=True) * 100

for label in ['合理诉求', '诉求偏高', '严重超额']:
    count = risk_counts.get(label, 0)
    percentage = risk_percentages.get(label, 0)
    print(f"  {label}: {count} 条记录 ({percentage:.2f}%)")

# 保存结果
output_filename = '预测结果_带风险标注.csv'
predict_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n8. 保存结果到: {output_filename}")

# 显示保存信息
import os
print(f"  文件大小: {os.path.getsize(output_filename) / 1024:.2f} KB")
print(f"  记录总数: {len(predict_df)}")
print(f"  列数: {len(predict_df.columns)}")

# 显示前几条记录的关键信息
print("\n前5条记录的风险标注结果:")
display_cols = ['索赔金额', '预测赔付金额', '理赔差额', '数分位指标', '风险标注']
if all(col in predict_df.columns for col in display_cols):
    print(predict_df[display_cols].head())

print("\n" + "=" * 60)
print("原始预测结果风险标注完成！")
print("=" * 60)
