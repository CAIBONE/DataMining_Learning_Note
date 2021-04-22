---
.title: Task1 赛题理解 date: 2021-04-21 20:27

---



# 一、学习知识点概要

[TOC]

# 二、学习内容

## 1.赛题分析

预测贷款申请人违约的可能.  

建模，用 80w 条数据的训练集训练模型，对 20w 条数据的 testA测试集 进行预测得到违约的概率. 

竞赛采用AUC作为评价指标.

## 2.预测结果评分指标

### 2.1 混淆矩阵

- $TP$（True Positive，真正例）预测为正例，实际也为正例.
- $FN$（False Negative，假负例）预测为负例，但实际为正例.
- $FP$（False Positive，假正例）预测为正例，但实际为负例.
- $TN$（True Negative，真负例）预测为负例，实际也为负例.

<div style="text-align: center">
<table>
<tr >
<td rowspan="2">真实情况</td>
<td colspan="2">预测结果</td>
</tr>
<tr >
<td>正例</td>
<td>反例</td>
</tr>
<tr>
<td>正例</td>
<td>TP</td>
<td>FN</td>
</tr>
<tr>
<td>负例</td>
<td>FP</td>
<td>TN</td>
</tr>
</table>
</div>


### 2.2 错误率与精度

- 错误率$E$（分类错误的样本数占样本总数的比率） 
  $$
  E=\frac{FT+FN}{TP+TN+FP+FN}
  $$

- 精度$acc$（分类正确的样本数占样本总数的比率）
  $$
  acc=\frac{TP+TN}{TP+TN+FP+FN}=1-E
  $$

### 2.3 *P-R*曲线与*F*度量

- $P\text{-}R$曲线 

  在很多情形下，我们可以根据学习器的预测结果对样例进行排序，排在前面的是学习器认为“最可能”是正例的样本，排在最后的则是学习器认为“最不可能”是正例的样本. 按此顺序逐个把样本作为正例进行预测，则每次可以计算出当前的查全率、查准率. 以查准率为纵轴、查全率为横轴作图，就得到了查准率-查全率曲线.简称“$P\text{-}R$曲线”.
  - 查准率$P$（precision，正确预测为正样本的数量$TP$占预测为正样本的数量$TP+FP$的比率） 

  $$
  P=\frac{TP}{TP+FP}
  $$

  - 查全率$R$（recall，正确预测为正样本的数量$TP$占正样本的数量$TP+FN$的比率） 
  $$
  R=\frac{TP}{TP+FN}
  $$
  ​	查准率和查全率是一对矛盾的度量. 一般来说，查准率高时，查全率往往偏低；而查全率高时，查准率又偏低.

  $P\text{-}R$图直观地显示出学习器在样本总体上的查全率、查准率. 在进行比较时，若一个学习器的曲$P\text{-}R$曲线被另一个学习器的曲线完全“包住”，则可断言后者性能更优. 但当两个学习器的$ROC$曲线发生交叉时，则难以一般性地断言两者孰优孰劣. 但人们往往仍希望两个学习器比出个高低，这时一个比较合理的判据是$P\text{-}R$曲线下的面积的大小. 但这个值不太好估算，因此人们设计了“平衡点”进行度量：

  - 平衡点（Break-Even Point，简称BEP）

    查准率=查全率时的取值. 取值越大，学习器性能更优. 

  但BEP还是过于简化了些，更常用的是下述的$F1$
  $$
  F1=\frac{2}{\frac{1}{P}+\frac{1}{R}}
  $$

  $$
  \frac{1}{F1}=\frac{1}{2}\cdot(\frac{1}{P}+\frac{1}{R})
  $$

  ​		$F1$的更一般形式——$F_\beta$
  $$
  \frac{1}{F_\beta}=\frac{1}{1+\beta^2}\cdot(\frac{1}{P}+\frac{\beta^2}{R})
  $$
  ​		$\beta>0$度量了查全率对查准率的相对重要性. $\beta=1$时退化为标准的$F1$；$\beta>1$时查全率有更大影响；$\beta<1$时查准率有更大影响.

### 2.4 ROC 和 AUC

- $ROC$ (Receive Operating Characteristics)
  根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算 TPR 和 FPR，分别以它们为横、纵坐标作图，就得到了“ ROC曲线 ”，显示该曲线的图称为“ROC图”.

    - $TPR$ (True Positive Rate，真正例率)

      在所有实际为正例的样本中，被正确地判断为正例的比率. 
  $$
      TPR=\frac{TP}{TP+FN}
  $$
  

    - $FPR$ (False Positive Rate，假正例率)
  
      在所有实际为负例的样本中，被正确地判断为正例的比率.
  
  $$
  FPR=\frac{FP}{FP+TN}
  $$
  
  当曲线越靠近左上角（代表学习器拥有较低假正例率的同时拥有较高真正例率），学习器性能越佳.
  如同$P\text{-}R$曲线一般，若一个学习器的$ROC$曲线被另一个学习器的曲线完全“包住”，则可断言后者性能更优. 但当两个学习器的$ROC$曲线发生交叉时，则难以一般性地断言两者孰优孰劣. 此时如果一定要比较，较为合理的判据是比较两者曲线下的面积，即下述的$AUG$.
  
- $\pmb{AUC}$ (Area Under Curve)

  被定义为$ROC$曲线下与坐标轴围成的面积:
  
  $$
  AUG=\int_0^1 ROC
  $$
  由于现实任务中通常是利用有限个测试样例来绘制$ROC$曲线，此时仅能获得有限个（真正例率，假正例率）坐标对，无法绘制出光滑的$ROC$曲线. 假定$ROC$曲线上由坐标为$\lbrace(
  x_1,y_1),(x_2,y_2)\ldots(x_m,y_m)\rbrace$的点按序连接而成$(x_1=0,x_m=1)$，则$AUG$可估算为： 
  $$
  AUG=\frac{1}{2}\sum_{i=1}^{m-1}(x_{i+1}-x_i)\cdot(y_i+y_{i+1})
  $$
   一般地$ROC$曲线处于直线$y=x$上方，故$AUG\in(0.5,1.0)$. $AUG$越大，学习机的性能越佳.

# 三、学习问题与解答


## 代码实现方法

### 1.*P-R*曲线与*F*1
```python
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
f1=f1_score(y_true,y_score)
```
### 2.*ROC*曲线与AUC
```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]
fpr, tpr, thersholds = roc_curve(y_true, y_score)
auc=roc_auc_score(y_true,y_score)
```
# 四、学习思考与总结

1、ROC曲线由于兼顾正例与负例，所以适用于评估分类器的整体性能，相比而言PR曲线完全聚焦于正例。
2、如果有多份数据且存在不同的类别分布，比如信用卡欺诈问题中每个月正例和负例的比例可能都不相同，这时候如果只想单纯地比较分类器的性能且剔除类别分布改变的影响，则ROC曲线比较适合，因为类别分布改变可能使得PR曲线发生变化时好时坏，这种时候难以进行模型比较；反之，如果想测试不同类别分布下对分类器的性能的影响，则PR曲线比较适合。
3、如果想要评估在相同的类别分布下正例的预测情况，则宜选PR曲线。
4、类别不平衡问题中，ROC曲线通常会给出一个乐观的效果估计，所以大部分时候还是PR曲线更好。
5、最后可以根据具体的应用，在曲线上找到最优的点，得到相对应的precision，recall，f1 score等指标，去调整模型的阈值，从而得到一个符合具体应用的模型。