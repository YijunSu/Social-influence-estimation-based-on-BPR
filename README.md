# Social Influence Estimation based on BPR

## Description

This is my graduation project.

The framework I used is pycharm + anaconda. It's just amazing!

When calculate the AUC benchmark I used the Leave One Out strategy.

BPR is the core model while other files are designed to deal with data.

Some of sql operation in the project has been hidden. 
Of course it's not necessary because the data has been stored in the text files. 


---

## Requests

### Environment
* python2.7

### Packages
* sklearn
* pymysql
* numpy
* scipy
* matplotlib

## Steps

### step1
Clone the repository to your local address.
```bash
git clone https://github.com/igoingdown/Social-influence-estimation-based-on-BPR.git
```

### step2
Train the model, test it and generate the AUC and MAP benchmark.
```bash
python bprWithEventNeighbor.py
```

### step3
Draw the AUC and MAP curve of BPR model comparing to the baselines.
```bash
python plotFigure.py
python plotMAPfigure.py
```

---

## 2017.6
> modified by **王雅轩**

* 解决代码bug

代码共有两个版本，对训练集和测试集的划分方式不同。
版本一：对于一个活动，将其一个用户信息作为测试集，其余均为训练集。
存在的问题：对训练集测试集和划分不符合一般机器学习的常规做法。

版本二：对于一个活动，将用户信息二八分成测试集和训练集。
存在的问题：学长代码的旧版本存在bug，需要调通。
解决方案：修改版本二的代码。发现由于在分集时引入了随机性，可能导致训练集或测试集为空集，
从而导致后续代码出错。在随机二八分数据集之前，
将其中随即两个用户信息随机分别放入训练集和测试集中，保证其非空性。

* 调参
1.	对参数规模进行调整

| num_factor | auc | map |
|:-----:|:-----:|:-----:|
| 10  | 0.781833778431 | 0.223724663189  |
| 20  | 0.792901661081 | 0.2351280467941 |
| 30  | 0.77632646694  | 0.221371891471  |
| 40  | 0.783558316081 | 0.23190639229   |
| 50  | 0.777131222483 | 0.218609806306  |

```
1.num_factor=20时效果最佳
2.根据对loss的观察，在迭代次数到达一定大小后，loss不再下降而是在一定范围内动荡，
  且最终的loss大约在550左右。因此，可通过调整其他参数降低loss的最终值以获得更好的效果。

```

2. 对学习速率进行调整

| learning_rate | auc | map |
|:-----:|:-----:|:-----:|
| 0.01  |  0.792901661081 | 0.2351280467941 |
| 0.05  |  0.784347899275 | 0.2303636939072 |
| 0.1   |  0.791392122752 | 0.242687040118 |

```
结论：迭代次数足够大时，learning_rate对效果影响不大。
推测：在迭代次数足够大的条件下，loss最终值趋向于相同，因此对效果影响不大。

```

* 主办方的近邻信息

==对原有的数据进行筛选，选取举办活动最多的主办方所举办的活动作为数据集（筛选数据时要注意去除那些没有用户参加信息的活动）
同样对参数规模进行调整==


| num_factor | auc | map |
|:-----:|:-----:|:-----:|
|10 | 0.851791011492 | 0.397193231371
|20 | 0.864155453935 | 0.371472650144
|30 | 0.857244614832 | 0.378085681617
|40 | 0.85171079043  |  0.360915963843
|50 | 0.858898226653 | 0.352393249684




















