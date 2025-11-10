# 折外（OOF）计算技术说明

## 1. 背景与目的
在使用聚合/目标编码类特征（如“城市/人员风险特征”“风险标注均值”等）时，若直接在全量训练集上聚合，再用于同一训练样本，会将该样本的标签信息“泄露”到其特征，从而导致评估偏乐观。折外（Out-Of-Fold, OOF）计算通过交叉验证框架，仅用“未包含该样本”的训练数据来生成该样本的聚合特征，有效避免数据泄露。

## 2. 定义
OOF 指在 K 折交叉验证中：对第 i 折样本，仅用其余 K-1 折训练数据计算聚合统计（均值、比例、计数、目标编码等），再将这些统计映射回第 i 折；最终拼接 K 次折外结果，得到训练集上的“无泄露”特征。

## 3. 正确实现流程
- 划分训练集为 K 折（常用 K=5）。
- 对每一折 i：
  - 用其余 K-1 折计算所需聚合表（按键/组合键分组，统计均值/比例/计数/目标编码等）。
  - 将该聚合表仅回填至第 i 折样本。
- 拼接各折结果，形成训练集的 OOF 特征。
- 用“全量训练集”再次聚合得到“推理用聚合表”，仅在预测阶段查表，绝不使用测试标签或测试样本参与重新统计。

## 4. 适用场景
- 目标均值编码/标签比例类特征。
- 本项目中的“始发/目的城市_风险特征”“寄/收件人_风险特征”以及四列“*_风险标注均值”。

## 5. 常见错误与规避
- 错误：全量训练集直接聚合并用于同一训练样本（自信息泄露）。
- 错误：在调参/交叉验证内对验证折使用了包含该折样本的聚合统计。
- 正确：严格 OOF；任何聚合统计与该样本的标签解耦。

## 6. 与阈值后处理的关系
阈值后处理（如确定 minority 概率和阈值）应基于训练/验证集完成，不能在测试集上搜索。OOF 负责消除特征层面的泄露；阈值后处理负责与评估指标（macro-F1）对齐，二者互补且均需“只在训练/验证数据上确定”。

## 7. 参考伪代码
```python
from sklearn.model_selection import StratifiedKFold

def build_oof_feature(df_train, key_col, target_col, how='mean', k=5, random_state=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    oof = pd.Series(index=df_train.index, dtype=float)
    for tr_idx, va_idx in skf.split(df_train, df_train[target_col]):
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]
        if how == 'mean':
            mapping = tr.groupby(key_col)[target_col].mean()
        elif how == 'rate':
            mapping = tr.groupby(key_col)[target_col].apply(lambda s: (s==1).mean())
        else:
            raise ValueError('Unsupported agg')
        oof.iloc[va_idx] = va[key_col].map(mapping)
    return oof

# 推理阶段聚合表（仅基于全量训练集）
def build_infer_mapping(df_train, key_col, target_col, how='mean'):
    if how == 'mean':
        return df_train.groupby(key_col)[target_col].mean()
    elif how == 'rate':
        return df_train.groupby(key_col)[target_col].apply(lambda s: (s==1).mean())
    else:
        raise ValueError('Unsupported agg')
```

## 8. 合规性声明
- 训练阶段：OOF 特征仅用训练集生成；每个样本的特征不包含自身标签信息；未使用测试集信息。
- 预测阶段：只读“训练集生成的聚合表”进行映射；对未知键按固定规则缺省处理；不在测试集上重新统计或调参。

以上流程确保折外计算不产生数据泄露，符合建模竞赛通行规范。


