# 项目文件说明

## 📁 核心文件结构

### 📓 Notebook文件（3个）

1. **数据预处理.ipynb** - 数据预处理流程
  
  - 功能：对test.csv进行预处理，生成test_processed.csv
  - 依赖：test.csv
  - 输出：test_processed.csv
2. **风险标注进阶.ipynb** - 模型训练流程
  
  - 功能：训练风险评估模型，包含特征工程、采样、参数搜索、阈值优化
  - 依赖：train_risked.csv
  - 输出：solution1p_fast.pkl（模型文件）
3. **预测.ipynb** - 预测流程
  
  - 功能：对测试集进行风险预测
  - 依赖：test_processed.csv, 聚合特征增强版数据集.csv, solution1p_fast.pkl
  - 输出：risk_predictions.csv（最终预测结果）

---

### 📊 数据文件（7个）

#### 原始数据

1. **train_risked.csv** - 训练数据（11,167条记录）
  
  - 用途：模型训练
  - 被使用于：风险标注进阶.ipynb
2. **test.csv** - 原始测试数据（2,792条记录）
  
  - 用途：数据预处理输入
  - 被使用于：数据预处理.ipynb

#### 中间数据

3. **test_processed.csv** - 预处理后的测试数据
  
  - 用途：预测流程的输入
  - 生成：数据预处理.ipynb
  - 被使用于：预测.ipynb (Cell 0)
4. **聚合特征增强版数据集.csv** - 历史聚合特征数据
  
  - 用途：提供历史风险特征用于预测
  - 生成：风险标注进阶.ipynb（特征工程部分）
  - 被使用于：预测.ipynb (Cell 0)
5. **test_with_hist_risk.csv** - 合并历史特征后的测试数据
  
  - 用途：预测流程的输入（包含历史特征）
  - 生成：预测.ipynb (Cell 0)
  - 被使用于：预测.ipynb (Cell 1)

#### 输出数据

6. **risk_predictions.csv** - 最终预测结果
  - 用途：提交结果文件
  - 生成：预测.ipynb (Cell 1)
  - 内容：包含预测类别、标签、概率等

---

### 🤖 模型文件（1个）

1. **solution1p_fast.pkl** - 训练好的LightGBM模型
  - 包含：Pipeline（BorderlineSMOTE + TomekLinks + LGBMClassifier）
  - 用途：预测
  - 被使用于：预测.ipynb (Cell 1)
  - 阈值：0.67（硬编码在预测代码中）

---

### 📄 文档文件（3个）

1. **风险评估模型技术路线文档.md** - 完整的技术路线说明
2. **数据预处理技术路线文档.md** - 数据预处理技术路线
3. **文件清理清单.md** - 文件清理记录（可删除）
4. **项目文件说明.md** - 本文件

---

## 🔄 完整工作流程

```
1. 数据预处理
   test.csv → [数据预处理.ipynb] → test_processed.csv

2. 模型训练
   train_risked.csv → [风险标注进阶.ipynb] → solution1p_fast.pkl + 聚合特征增强版数据集.csv

3. 预测
   test_processed.csv + 聚合特征增强版数据集.csv 
   → [预测.ipynb Cell 0] 
   → test_with_hist_risk.csv

   test_with_hist_risk.csv + solution1p_fast.pkl
   → [预测.ipynb Cell 1]
   → risk_predictions.csv（最终结果）
```

---

## ⚠️ 重要说明

1. **必需文件**：上述所有核心文件都是必需的，删除任何文件可能导致流程中断
2. **运行顺序**：
  - 先运行：数据预处理.ipynb
  - 再运行：风险标注进阶.ipynb（如果需要重新训练）
  - 最后运行：预测.ipynb
3. **模型阈值**：当前使用固定阈值0.67，如需使用训练时优化的阈值，需要生成model_balanced_optimized.pkl文件
