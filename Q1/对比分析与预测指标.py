# -*- coding: utf-8 -*-
"""
对比标准结果（训练数据风险标注）与预测数据风险标注，计算预测指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("标准结果与预测数据对比分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n1. 加载数据...")

# 加载标准结果（训练数据风险标注）
train_df = pd.read_csv('train_with_risk_annotation.csv')
print(f"标准结果（训练数据）: {train_df.shape}")
print(f"  记录数: {len(train_df)}")

# 加载预测数据风险标注
predict_df = pd.read_csv('我的预测结果_带风险标注.csv')
print(f"预测数据: {predict_df.shape}")
print(f"  记录数: {len(predict_df)}")

# ============================================================================
# 2. 风险标注分布对比
# ============================================================================
print("\n" + "=" * 80)
print("2. 风险标注分布对比")
print("=" * 80)

# 标准结果分布
train_risk_dist = train_df['风险标注'].value_counts().sort_index()
train_risk_pct = train_df['风险标注'].value_counts(normalize=True).sort_index() * 100

# 预测数据分布
predict_risk_dist = predict_df['风险标注'].value_counts().sort_index()
predict_risk_pct = predict_df['风险标注'].value_counts(normalize=True).sort_index() * 100

# 创建对比表
comparison_df = pd.DataFrame({
    '标准结果_数量': train_risk_dist,
    '标准结果_占比': train_risk_pct,
    '预测数据_数量': predict_risk_dist,
    '预测数据_占比': predict_risk_pct
}).fillna(0)

# 计算差异
comparison_df['数量差异'] = comparison_df['预测数据_数量'] - comparison_df['标准结果_数量']
comparison_df['占比差异'] = comparison_df['预测数据_占比'] - comparison_df['标准结果_占比']
comparison_df['占比差异百分比'] = (comparison_df['占比差异'] / comparison_df['标准结果_占比'] * 100).replace([np.inf, -np.inf], np.nan)

print("\n风险标注分布对比表:")
print(comparison_df.round(2))

# ============================================================================
# 3. 数分位指标统计对比
# ============================================================================
print("\n" + "=" * 80)
print("3. 数分位指标统计对比")
print("=" * 80)

# 标准结果数分位指标统计
train_stats = train_df.groupby('风险标注')['数分位指标'].describe().round(4)

# 预测数据数分位指标统计
predict_stats = predict_df.groupby('风险标注')['数分位指标'].describe().round(4)

print("\n标准结果（训练数据）数分位指标统计:")
print(train_stats)

print("\n预测数据数分位指标统计:")
print(predict_stats)

# 对比均值差异
print("\n数分位指标均值对比:")
mean_comparison = pd.DataFrame({
    '标准结果均值': train_stats['mean'],
    '预测数据均值': predict_stats['mean'],
    '均值差异': predict_stats['mean'] - train_stats['mean'],
    '相对差异(%)': ((predict_stats['mean'] - train_stats['mean']) / train_stats['mean'] * 100)
}).round(4)
print(mean_comparison)

# ============================================================================
# 4. 关键指标对比
# ============================================================================
print("\n" + "=" * 80)
print("4. 关键指标对比")
print("=" * 80)

# 定义关键指标列
key_cols = ['索赔金额', '理赔差额', '数分位指标']
if '实际赔付金额' in train_df.columns:
    key_cols.insert(1, '实际赔付金额')
if '预测赔付金额' in predict_df.columns:
    key_cols.insert(1, '预测赔付金额')

# 过滤存在的列
train_cols = [col for col in key_cols if col in train_df.columns]
predict_cols = [col for col in key_cols if col in predict_df.columns]

print("\n标准结果（训练数据）各风险类别关键指标统计:")
train_summary = train_df.groupby('风险标注')[train_cols].agg(['mean', 'std', 'median']).round(2)
print(train_summary)

print("\n预测数据各风险类别关键指标统计:")
predict_summary = predict_df.groupby('风险标注')[predict_cols].agg(['mean', 'std', 'median']).round(2)
print(predict_summary)

# ============================================================================
# 5. 预测指标计算
# ============================================================================
print("\n" + "=" * 80)
print("5. 预测指标计算")
print("=" * 80)

# 5.1 分布一致性指标
print("\n5.1 分布一致性指标:")

# 计算KL散度（相对熵）作为分布差异度量
from scipy import stats

# 对齐风险类别
risk_labels = ['合理诉求', '诉求偏高', '严重超额']
train_probs = [train_risk_pct.get(label, 0) / 100 for label in risk_labels]
predict_probs = [predict_risk_pct.get(label, 0) / 100 for label in risk_labels]

# 归一化
train_probs = np.array(train_probs) / np.sum(train_probs)
predict_probs = np.array(predict_probs) / np.sum(predict_probs)

# 避免零值
train_probs = np.clip(train_probs, 1e-10, 1)
predict_probs = np.clip(predict_probs, 1e-10, 1)

# KL散度
kl_divergence = stats.entropy(predict_probs, train_probs)
print(f"  KL散度 (KL Divergence): {kl_divergence:.6f}")
print(f"    解释: 值越小表示分布越相似，0表示完全一致")

# 卡方检验（分布差异显著性检验）
observed = np.array([predict_risk_dist.get(label, 0) for label in risk_labels])
expected = np.array([train_risk_pct.get(label, 0) / 100 * len(predict_df) for label in risk_labels])
expected = np.clip(expected, 0.1, None)  # 避免零值

chi2_stat, p_value = stats.chisquare(observed, expected)
print(f"  卡方统计量: {chi2_stat:.4f}")
print(f"  p值: {p_value:.6f}")
print(f"    解释: p < 0.05 表示分布差异显著")

# 5.2 数分位指标对比
print("\n5.2 数分位指标对比:")

# 整体数分位指标统计
train_overall_mean = train_df['数分位指标'].mean()
train_overall_std = train_df['数分位指标'].std()
predict_overall_mean = predict_df['数分位指标'].mean()
predict_overall_std = predict_df['数分位指标'].std()

print(f"  标准结果整体统计:")
print(f"    均值: {train_overall_mean:.4f}")
print(f"    标准差: {train_overall_std:.4f}")
print(f"  预测数据整体统计:")
print(f"    均值: {predict_overall_mean:.4f}")
print(f"    标准差: {predict_overall_std:.4f}")
print(f"  均值差异: {predict_overall_mean - train_overall_mean:.4f}")
print(f"  相对差异: {((predict_overall_mean - train_overall_mean) / train_overall_mean * 100):.2f}%")

# 5.3 各风险类别数分位指标差异
print("\n5.3 各风险类别数分位指标差异:")
for label in risk_labels:
    if label in train_df['风险标注'].values and label in predict_df['风险标注'].values:
        train_mean = train_df[train_df['风险标注'] == label]['数分位指标'].mean()
        predict_mean = predict_df[predict_df['风险标注'] == label]['数分位指标'].mean()
        diff = predict_mean - train_mean
        rel_diff = (diff / train_mean * 100) if train_mean != 0 else 0
        
        print(f"  {label}:")
        print(f"    标准结果均值: {train_mean:.4f}")
        print(f"    预测数据均值: {predict_mean:.4f}")
        print(f"    差异: {diff:.4f} ({rel_diff:+.2f}%)")

# 5.4 预测准确度指标（如果有可能）
print("\n5.4 分布匹配度指标:")
print(f"  合理诉求占比匹配度:")
train_reasonable = train_risk_pct.get('合理诉求', 0)
predict_reasonable = predict_risk_pct.get('合理诉求', 0)
print(f"    标准: {train_reasonable:.2f}%")
print(f"    预测: {predict_reasonable:.2f}%")
print(f"    差异: {predict_reasonable - train_reasonable:.2f}%")

print(f"  严重超额占比匹配度:")
train_excessive = train_risk_pct.get('严重超额', 0)
predict_excessive = predict_risk_pct.get('严重超额', 0)
print(f"    标准: {train_excessive:.2f}%")
print(f"    预测: {predict_excessive:.2f}%")
print(f"    差异: {predict_excessive - train_excessive:.2f}%")

# 5.5 预测质量评估
print("\n5.5 预测质量评估:")

# 计算预测覆盖率（预测数据是否覆盖了所有风险类别）
coverage = len([label for label in risk_labels if label in predict_df['风险标注'].values]) / len(risk_labels) * 100
print(f"  风险类别覆盖率: {coverage:.1f}%")

# 计算分布平衡度（标准差越小越平衡）
dist_balance = 1 / (1 + predict_risk_pct.std())  # 归一化到0-1
print(f"  分布平衡度: {dist_balance:.4f} (接近1表示分布更均匀)")

# ============================================================================
# 6. 可视化对比
# ============================================================================
print("\n" + "=" * 80)
print("6. 生成可视化对比图表...")
print("=" * 80)

# 创建对比图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 风险标注分布对比柱状图
ax1 = axes[0, 0]
x = np.arange(len(risk_labels))
width = 0.35

train_counts = [train_risk_dist.get(label, 0) for label in risk_labels]
predict_counts = [predict_risk_dist.get(label, 0) for label in risk_labels]

bars1 = ax1.bar(x - width/2, train_counts, width, label='标准结果', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, predict_counts, width, label='预测数据', alpha=0.8, color='coral')

ax1.set_xlabel('风险类别', fontsize=12)
ax1.set_ylabel('记录数', fontsize=12)
ax1.set_title('风险标注分布对比（数量）', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(risk_labels)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 6.2 风险标注占比对比柱状图
ax2 = axes[0, 1]
train_pcts = [train_risk_pct.get(label, 0) for label in risk_labels]
predict_pcts = [predict_risk_pct.get(label, 0) for label in risk_labels]

bars3 = ax2.bar(x - width/2, train_pcts, width, label='标准结果', alpha=0.8, color='steelblue')
bars4 = ax2.bar(x + width/2, predict_pcts, width, label='预测数据', alpha=0.8, color='coral')

ax2.set_xlabel('风险类别', fontsize=12)
ax2.set_ylabel('占比 (%)', fontsize=12)
ax2.set_title('风险标注分布对比（占比）', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(risk_labels)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# 6.3 数分位指标分布对比箱线图
ax3 = axes[1, 0]
box_data_all = []
box_labels_all = []

for label in risk_labels:
    train_data = train_df[train_df['风险标注'] == label]['数分位指标'].values
    if len(train_data) > 0:
        box_data_all.append(train_data)
        box_labels_all.append(f'{label}\n标准')
    
    predict_data = predict_df[predict_df['风险标注'] == label]['数分位指标'].values
    if len(predict_data) > 0:
        box_data_all.append(predict_data)
        box_labels_all.append(f'{label}\n预测')

if box_data_all and len(box_data_all) == len(box_labels_all):
    bp = ax3.boxplot(box_data_all, tick_labels=box_labels_all, patch_artist=True)
    colors = ['lightblue'] * 3 + ['lightcoral'] * 3
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('数分位指标', fontsize=12)
    ax3.set_title('数分位指标分布对比（箱线图）', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

# 6.4 数分位指标均值对比
ax4 = axes[1, 1]
train_means = [train_stats.loc[label, 'mean'] if label in train_stats.index else 0 for label in risk_labels]
predict_means = [predict_stats.loc[label, 'mean'] if label in predict_stats.index else 0 for label in risk_labels]

bars5 = ax4.bar(x - width/2, train_means, width, label='标准结果', alpha=0.8, color='steelblue')
bars6 = ax4.bar(x + width/2, predict_means, width, label='预测数据', alpha=0.8, color='coral')

ax4.set_xlabel('风险类别', fontsize=12)
ax4.set_ylabel('数分位指标均值', fontsize=12)
ax4.set_title('各风险类别数分位指标均值对比', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(risk_labels)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('预测指标对比分析.png', dpi=300, bbox_inches='tight')
print("  图表已保存为: 预测指标对比分析.png")
plt.close()

# ============================================================================
# 7. 保存对比结果
# ============================================================================
print("\n" + "=" * 80)
print("7. 保存对比结果")
print("=" * 80)

# 保存详细的对比结果
output_data = {
    '风险类别': risk_labels,
    '标准结果_数量': [train_risk_dist.get(label, 0) for label in risk_labels],
    '标准结果_占比(%)': [train_risk_pct.get(label, 0) for label in risk_labels],
    '预测数据_数量': [predict_risk_dist.get(label, 0) for label in risk_labels],
    '预测数据_占比(%)': [predict_risk_pct.get(label, 0) for label in risk_labels],
    '数量差异': [predict_risk_dist.get(label, 0) - train_risk_dist.get(label, 0) for label in risk_labels],
    '占比差异(%)': [predict_risk_pct.get(label, 0) - train_risk_pct.get(label, 0) for label in risk_labels]
}

comparison_result = pd.DataFrame(output_data)
comparison_result.to_csv('预测指标对比结果.csv', index=False, encoding='utf-8-sig')
print("  对比结果已保存为: 预测指标对比结果.csv")

# ============================================================================
# 8. 总结报告
# ============================================================================
print("\n" + "=" * 80)
print("8. 预测指标总结报告")
print("=" * 80)

print(f"\n【分布一致性】")
print(f"  KL散度: {kl_divergence:.6f}")
print(f"  卡方统计量: {chi2_stat:.4f}, p值: {p_value:.6f}")
if p_value > 0.05:
    print(f"  ✓ 分布差异不显著（p > 0.05）")
else:
    print(f"  ✗ 分布差异显著（p < 0.05）")

print(f"\n【数分位指标】")
print(f"  标准结果均值: {train_overall_mean:.4f}")
print(f"  预测数据均值: {predict_overall_mean:.4f}")
print(f"  均值差异: {predict_overall_mean - train_overall_mean:.4f} ({((predict_overall_mean - train_overall_mean) / train_overall_mean * 100):.2f}%)")

print(f"\n【分布匹配度】")
print(f"  合理诉求占比匹配: {abs(predict_reasonable - train_reasonable):.2f}% 差异")
print(f"  严重超额占比匹配: {abs(predict_excessive - train_excessive):.2f}% 差异")

print(f"\n【预测质量】")
print(f"  风险类别覆盖率: {coverage:.1f}%")
print(f"  分布平衡度: {dist_balance:.4f}")

print("\n" + "=" * 80)
print("对比分析完成！")
print("=" * 80)
