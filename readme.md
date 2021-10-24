# cs 224n

## Lecture 1  wordvecs1

### Slides:

#### 1.一些想法：

- 就像人是由周围的社会关系决定的，词是由上下文决定的。(PS：如果用马原来指导 这不是显然的嘛：） 把词当作人 人是一切社会关系的总和 词是一切上下文关系的总和 )

- 一个词的表达有多了解他所处的周围坏境(及他的社会关系) 他就有多了解他自己 就有多接近他本身的意思。

#### 2. 关于点积的疑惑点：

[Mark] <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210903110729283.png" alt="image-20210903110729283" style="zoom:50%;" />

为什么 点积越大与相似呢 比如说A,B 只有在A+B=C(定值) 时 A=B时点积最大 但是在这里 ui vi的和并不是一定的呀

为什么不直接用两个向量的差的向量的2-范数来表示？ 而且还是恒正的。 是因为计算量大嘛？

也许可以从向量内积的集合意义上入手理解：

> 内积的几何意义 
>
> 点乘的几何意义是可以用来表征或计算两个向量之间的夹角，以及在b向量在a向量方向上的投影。
>
> 那么归一化后 就是在超球上的两个向量的夹角的cos值 该值越大 夹角越小 相似度越大
>
> Cosine Distance：  1 - Cosine Similarity   余弦距离



#### 3.softmax理解摘要:

给定一组数据 或者可以看作一个向量  用softmax可以给出他们的一个概率分布：

<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210903111751793.png" alt="image-20210903111751793" style="zoom:33%;" />

max:是因为最大的那个数据的概率(相比与直接平均)

soft:只因为仍然给那些小的数据一定的概率

这个挺漂亮的 但是最大的症结是为什么 ui*vi 越大 两个向量越相似。[**已解决(见2.)**]






### Suggested Readings:

#### 1.softax函数的一个性质：

对于softmax函数 如果所有的xi进行同样程度的平移(即减去一个向量) 函数输出不变。也就是说，它有一组“冗余”的参数(即可以将其中一个xi变为0)。 

#### 2. softmax 与 logistic regression:

softmax函数当其K=2时 依据1进行变形就会变为logistic regression:
<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210907144235555.png" alt="image-20210907144235555" style="zoom:67%;" />

#### 3. 解决模型过大 

运行梯度下降慢 过度拟合的问题 有两种既加快训练速度又提升词向量质量的办法：

- 对频繁词进行二次采样以减少训练示例的数量
- 使用“负采样”的技术修改优化目标，每个训练样本仅更新模型权重的一小部分。

#### 4. 二次采样：

word2vec C 代码实现了一个公式，用于计算在词汇表中保留给定单词的概率：

<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210907145855001.png" alt="image-20210907145855001" style="zoom: 50%;" />

z(wi) 是语料库中属于该词的总词的概率分数

- 当 z(wi)<=0.0026z时，P(wi)=1.0P(wi)=1.0（被保留的机会为 100%）这意味着只有占总单词 0.26% 以上的单词才会被子采样。
- 当 z(wi)=0.00746时，P(wi)=0.5P(wi)=0.5（50% 的机会被保留）。
- 当 z(wi)=1.0，P(wi)=0.033P(wi)=0.033（3.3% 的机会被保留）。 也就是说，如果语料库完全由wiwi这个词组成，荒谬。

> [论文中定义的这个函数与 C 代码中实现的略有不同，但认为 C 实现是更权威的版本]

#### 5.负采样:

- 思路：例如，在词对（“fox”、“quick”）上训练网络时，网络的“标签”或“正确输出”是一个单热向量。也就是说，对应于“quick”的输出神经元输出一个 1，而所有其他数千个输出神经元输出一个 0。 使用负采样，将随机选择少量“负”词（假设为 5 个）来更新权重。 （因此，只更新"正"词（“quick”）的权重，加上想要输出 0 的其他 5 个词的权重。总共有 6 个输出神经元，总共 1,800 个权重值。只是输出层中 3M 权重的 0.06%！

- 选择负样本： “负样本”（即我们将训练输出 0 的 5 个输出词）使用“一元分布”选择，其中更频繁的词更有可能被选为负样本。即利用单词出现的概率作为选择概率。作者在他们的论文中尝试了对该等式的多种变体，其中表现最好的是将字数提高到 3/4 次方：

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210907151509564.png" alt="image-20210907151509564" style="zoom:25%;" />

  该等式倾向于增加出现频率较低的词的概率并降低出现频率较高的词的概率

> [论文称，选择 5-20 个单词对于较小的数据集效果很好，而对于大型数据集，只需选择 2-5 个单词即可。]

> [在隐藏层中，仅更新输入词的权重（无论是否使用负采样都是如此）]



## Assignment 1: 

#### 1.奇异值分解(SVD):

[https://zhuanlan.zhihu.com/p/29846048]

[https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf]

[lectures [7](https://web.stanford.edu/class/cs168/l/l7.pdf), [8](http://theory.stanford.edu/~tim/s15/l/l8.pdf), and [9](https://web.stanford.edu/class/cs168/l/l9.pdf) of CS168 课程笔记提供了对压缩通用算法(PCA/SVD)的高级处理]

SVD是对数据进行有效特征整理的过程。首先，对于一个m×n矩阵A，可以理解为其有m个数据，n个特征，（想象成一个n个特征组成的坐标系中的m个点），然而一般情况下，这n个特征并不是正交的，也就是说这n个特征并不能归纳这个数据集的特征。

SVD的作用就相当于是一个坐标系变换的过程，从一个不标准的n维坐标系，转换为一个标准的k维坐标系，并且使这个数据集中的点，到这个新坐标系的欧式距离为最小值（也就是这些点在这个新坐标系中的投影方差最大化），其实就是一个最小二乘的过程。

进一步，如何使数据在新坐标系中的投影最大化呢，就需要让这个新坐标系中的基尽可能的不相关，可以用协方差来衡量这种相关性。当对这个协方差矩阵进行特征分解之后，可以得到奇异值和右奇异矩阵，而右奇异矩阵则是一个新的坐标系，奇异值则对应这个新坐标系中每个基对于整体数据的影响大小，这时便可以提取奇异值最大的k个基作为新的坐标。

#### 2.外积、内积和叉积：

英语语境里，外积(outer)和叉积(cross)是不一样的，外积是列向量乘行向量（内积相反），叉积是叉乘的结果。

#### 3.matplotlib.pyplot.imshow 的一个小问题：

matplotlib.pyplot.imshow 的一个小问题是它可能会产生奇怪的结果(如果呈现的数据不是 uint8) 为了解决这个问题，应该在显示之前将图像显式转换为 uint8。

```python
plt.imshow(np.uint8(img_tinted))
plt.show()
```

#### 4. 高效SVD：

numpy、scipy 和 scikit-learn（sklearn）都提供了一些 SVD 的实现，但只有 scipy 和 sklearn 提供了 Truncated SVD 的实现，只有 sklearn 提供了计算大规模 Truncated SVD 的高效随机算法。所以应该使用 sklearn.decomposition.TruncatedSVD。

#### 5. SVD实现的小细节：

SVD 存在“符号不确定性”的问题，这意味着组件的符号和变换的输出取决于算法和随机状态。要解决此问题，应将此类的实例与数据拟合一次，然后保留该实例以进行转换。

> https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

```python
SVD=TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced=SVD.fit_transform(M)
```

#### 6. numpy broadcasting :

> [Computation on Arrays: Broadcasting by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).

#### 7. 绘制图的参考 可以参考matplotlib库 简直应有尽有 ：

> [the Matplotlib gallery](https://matplotlib.org/gallery/index.html)

#### 8. Word2Vec使用小细节：

- 没法用gensim.downloader.load() 改成本地下载

  ```python
  from gensim import models
  
  wv_from_bin = models.KeyedVectors.load_word2vec_format(
      '/home/wangxidong/gensim-data/word2vec-google-news-300/GoogleNews-vectors-negative300.bin', binary=True)
  
  -vocab = list(wv_from_bin.vocab.keys())
  -print("Loaded vocab size %i" % len(vocab))
  ```

- 但上述方法的倒数第二行在gensim更新为4.0.0即以上后无法使用 

  > 参考https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

  使用如下方法代替:

  ```python
   vocab = list(wv_from_bin.index_to_key)
  ```

#### 9. 查conda库：

> https://anaconda.org/



## Lecture 2 wordvecs2

### Slides:

#### 1. entropy:

- 计算事件的信息:

  信息论背后的基本直觉是，了解不太可能发生的事件比了解可能发生的事件提供更多信息。 

  > information(x) = h(x) = -log( p(x) )  log以2为底  所以衡量信息信息度量的单位是比特

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915213548739.png" alt="image-20210915213548739" style="zoom: 25%;" />

- 计算随机变量的熵：

  熵 表示或传输 从随机变量的概率分布中 提取事件 所需的平均比特数

  > H(X) = -sum(each k in K p(k) * log(p(k)))

  最低熵是针对具有概率为 1.0（确定性）的单个事件的随机变量计算的。 随机变量的最大熵是所有事件的可能性相等。

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915213524961.png" alt="image-20210915213524961" style="zoom: 25%;" />

  > 注意，在计算熵时，必须为概率添加一个很小的值，以避免计算零值的对数，这会导致无穷大而不是数字。

#### 2. cross entropy：

交叉熵是信息论领域的一种度量，建立在熵的基础上，是对给定随机变量或事件集的两个概率分布之间差异的度量。

**交叉熵是当使用模型 q 时，对来自分布为 p 的源的数据进行编码所需的平均位数**

- 两个概率分布之间的交叉熵，例如来自 P 的 Q，可以正式表示为：

  > H(P, Q) = – sum x in X P(x) * log(Q(x))

  其中 P 可能是目标分布，Q 是目标分布的近似值。

  其中 P(x) 是事件 x 在 P 中的概率，Q(x) 是事件 x 在 Q 中的概率，log 是以 2 为底的对数，这意味着结果以位为单位。

  [如果改为使用 base-e 或自然对数，则结果的单位将称为 nats]

  如果两个概率分布相同，结果将是一个以比特为单位测量的正数，并且将等于分布的熵。

  



### Suggested Readings:

#### 1.GloVe: Global Vectors for Word Representation 

 https://nlp.stanford.edu/pubs/glove.pdf

##### (1)技术创新点与此想法背后的产生逻辑

- 作者看到了Mikolov的Word2vec的模型 相较于传统的co-occurence 对数据利用不充分的问题 进行突破

- 作者想结合两个模型 所以就涉及到删减 所以作者先探究是什么造就的核心优势表现(如本文中 Word2vec是如何学习到多维度的规律向量的)  然后把这几个数学核心进行再加工与组合  

  下面就是想将借鉴co-occurrence的思想优化Word2Vec模型中的向量初始化部分

- 词向量 与 词对频率 关系的  寻找过程：
  $$
  X_{ij}:表示在单词 j 在单词 i 的上下文中出现的次数。 i是中心词
  $$
  

  - 基于比值思想与很好的实验结果找到初始函数<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915160954774.png" alt="image-20210915160954774" style="zoom: 40%;" />

  - 基于向量空间本质是线性结构 引入向量差<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915161100832.png" alt="image-20210915161100832" style="zoom:40%;" />

  - 观察到向量和标量的区别 引入内积 <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915161525563.png" alt="image-20210915161525563" style="zoom:40%;" />

  - 考虑到F必须为(R,+) 和 (R>0, ×) 之间的一个同态映射[F(g1 + g2) = F(g1)×F(g2)]  即类似<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915163840904.png" alt="image-20210915163840904" style="zoom:40%;" />

    引入log/exp  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915163555544.png" alt="image-20210915163555544" style="zoom:40%;" />

  - 考虑到中心词和上下文词的轮换对称性 将影响上述对称性的log(Xi)写为了中心词的偏移量bi  并为了保持轮换对称加入了探索词偏移量bk <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915165136436.png" alt="image-20210915165136436" style="zoom:50%;" />

  - 差不多了 但是有一点是若参数Xik为0对数会发散

    - 思路一：log(Xik ) → log(1 + Xik ) [避免发散的同时保持 X 的稀疏性]
    - 思路二：LSA(即SVD法) [主要缺点是它平等地权衡所有共现，即使那些很少发生或从不发生的]

    作者思路：采用最小二乘拟合<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915194002751.png" alt="image-20210915194002751" style="zoom:40%;" />

    f满足条件：

    - x趋于零时快速收敛(使得logX的平方有限)
    - f非递减 显然
    - 对于较大的 x 值，f (x) 应该相对较小 因为太大其实携带的信息量反而很小

    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915200651541.png" alt="image-20210915200651541" style="zoom:50%;" />

    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915200711026.png" alt="image-20210915200711026" style="zoom:67%;" />

- 基于对Word2Vec的代价函数的思考和先前思考 得出Glove代价函数：

  - 对于Word2Vec代价函数的变形：<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915225443768.png" alt="image-20210915225443768" style="zoom: 50%;" />

    [所用技巧：和项的重新组合 相关参数的替换 往一些熟悉的结果上靠(如cross entropy **看到log想一下凑一个概率**)   ]

  - 结合先前思考 得出：<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915231303012.png" alt="image-20210915231303012" style="zoom:40%;" />

    [所用技巧：从实际出发替换掉相似度评判指标 <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915231719149.png" alt="image-20210915231719149" style="zoom: 50%;" />    类似自己于lecture1 2的思考   :)   不过又加了个log

    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210915231836380.png" alt="image-20210915231836380" style="zoom: 50%;" />      简直神来之笔   最后把Xi替换为更普遍的权重函数f 前后至此全部打通 :)

#####  (2)写作词汇

- **garner substantial** support 获得大力支持

- **argue** 认为 (有点争论的味道)    **outperform** 超过    **domenstrate** 证明

- **opaque** 不透明

- **probe** the underlying  feature 探索潜在特征

- the efficiency **with which** the methods capture features can be **advantageous.**  但方法捕获特征的效率可能是有利的

- word **analogies** 词类比

- **suffer** significant **drawbacks** 面临重大弊端

-  **leverage** statistical information 利用数据信息

- The model perfroms well, **as evidenced by** its **state-of-the-art** performance 该模型表现良好，其最先进的性能证明了这一点

- In this section, we **shed some light on** this question.  在本节中，我们阐明了这个问题。

- We begin with a simple example that **showcases** how to do this 我们从一个简单的例子开始，展示了如何做这些

- distinguish/**discriminate** 区分

- some **as-of-yet** unspecified parameters  一些尚未指定的参数

- enforcing a few **desiderata** 强制执行一些要求

- **drastic** simplification 大幅简化

- A main **drawback** to this model is that  该模型的主要缺点是

- A **bears some resemblance** to B     A 与 B 有一些相似之处

##### (3)行文手法、结构与引人入胜点

- abstract:
  - 现有模型很好但是有一些缺点
  - 自己想法产生+数学大细节+核心细节创新点及表现好的原因+具体实验表现
- conclusion:
  - 目前大家的关注和一些争论(可以举例)
  - 本文的想法及原因
  - 本文模型的核心亮点和数学大细节(一句话)
  - 实验结果表现

- introduction:
  - 用技术的适用领域来进行论文的引用
  - 重点技术领域的分析(时间+技术本身)(可以举例)
  - 各类模型的分类及其优缺点(本模型想法的产生)
  - 本文模型的数学大细节+模型如何表现良好
- model:
  - 逐步讲清模型的数学推导过程 并讲清楚了动机与思考
  - 分析算法复杂度
- Experiments:
  - 评价方法
  - 数据库和训练细节
  - 结果
  - 模型分析
    - 向量长度和上下文大小
    - 语料库大小
    - 运行时间 
    - 与其他模型的对比 











## Python review session:

### Slides:

#### 1. python:

python-review.ipynb

https://www.w3schools.com/python/

#### 2. numpy:

https://cs231n.github.io/python-numpy-tutorial/

https://numpy.org/doc/stable/user/quickstart.html

#### 3. MATPLOTLIB:

https://matplotlib.org/stable/gallery/index.html







## Lecture 3  neural nets

### Slides:

#### 1. 一些基本知识





## Lecture 4 backprop

### Slides:

#### 1.矩阵运算技巧：

训练尽量vectorized 而不是 for 循环

#### 2. 非线性函数：

为了构建一个前馈深度网络，应该尝试的第一件事是 ReLU——由于良好的梯度回流，它可以快速训练并且表现良好

#### 3. [**Mark**]  参数初始化：

-  **通常必须将权重初始化为小的随机值  避免产生对称性**

- Xavier 初始化的方差与扇入 nin（前一层大小）和扇出 nout（下一层大小）成反比：

<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210922215346144.png" alt="image-20210922215346144" style="zoom: 33%;" />

#### 4. 最优化：

对于复杂的网络和情况，通常使用一系列更复杂的“自适应”优化器中的一个会做得更好，这些优化器通过累积梯度来缩放参数调整。

- Adagrad
- RMSprop
- Adam [在许多情况下，相当好的、安全的]
- SparseAdam
- …  

#### 5. 学习率

通常可以通过在训练时降低学习率来获得更好的结果

- 手动：每 k 个 epoch 将学习率减半

- 通过公式：<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210922221639273.png" alt="image-20210922221639273" style="zoom:33%;" /> 

  > epoch[对数据的一次传递(打乱或采样)]

- 还有更高级的方法，比如循环学习率 (q.v.) [帮助模型跳出大坑]

> 更高级的优化器仍然使用学习率，但它可能是优化器收缩的初始速率——因此可能能够从高开始(如0.1)



### Suggested Readings:

#### 1. backprop的一些细节：

注意梯度消失等一些问题



## Assignment 2



#### 1.矩阵求导参考：

[Review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/review-differential-calculus.pdf)

**a2 的 gradient-notes**

##### 1.1 几点注意事项：

- 区分 微分 梯度 偏导数

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926101952693.png" alt="image-20210926101952693" style="zoom: 50%;" />

- 注意在**Rn到R**的关系中(即只有一个输出函数/输出矩阵有一个维度为一维) Jacobian行列式与梯度之间有一个转置关系<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926102137208.png" alt="image-20210926102137208" style="zoom: 67%;" />

  因为定义的时候Jacobian就是个行向量<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926102533806.png" alt="image-20210926102533806" style="zoom: 50%;" />

  但是梯度被定义为列向量 所以之间有个转置的关系

  > 当有多个输出函数时(大多数情况)  两者数值上相同

- 标记<img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926104104114.png" alt="image-20210926104104114" style="zoom: 40%;" />符号通常是模棱两可的，可以指梯度或Jacobian行列式。







##### 1.2 几个有用的矩阵求导二级结论

(推导见a2 gradient-notes)

- 矩阵乘以列向量 并对列向量求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926110102480.png" alt="image-20210926110102480" style="zoom:50%;" />    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926110126823.png" alt="image-20210926110126823" style="zoom:38%;" />

- 行向量乘以矩阵 并对行向量求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210926110205589.png" alt="image-20210926110205589" style="zoom:50%;" />     <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927134743695.png" alt="image-20210927134743695" style="zoom:33%;" />

- 向量对自己求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927134825216.png" alt="image-20210927134825216" style="zoom:50%;" />     <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927134848084.png" alt="image-20210927134848084" style="zoom:33%;" />

- 作用在矩阵元素上的函数 对矩阵求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135024680.png" alt="image-20210927135024680" style="zoom:50%;" />     <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135045869.png" alt="image-20210927135045869" style="zoom:33%;" />

- 矩阵乘以列向量 并对矩阵求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135237137.png" alt="image-20210927135237137" style="zoom:25%;" />    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135323217.png" alt="image-20210927135323217" style="zoom:33%;" />

- 行向量乘以矩阵 并对矩阵求导

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135451924.png" alt="image-20210927135451924" style="zoom:50%;" />    <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927135523022.png" alt="image-20210927135523022" style="zoom:33%;" />

- 与 logits 相关的交叉熵损失 

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927201017591.png" alt="image-20210927201017591" style="zoom: 67%;"/> 

  <img src="C:\Users\wangxidong\AppData\Roaming\Typora\typora-user-images\image-20210927201053330.png" alt="image-20210927201053330" style="zoom:33%;" />  (如果为y是列向量的话 就转置一下)

  <img src="D:\cs224n\笔记图片\扫描全能王 2021-09-27 20.14.jpg" alt="扫描全能王 2021-09-27 20.14" style="zoom: 33%;" />

- 矩阵求导如果没有square型的矩阵 则只有唯一的满足矩阵乘法性质的排列和转置 所以可以用这个性质去检验

#### 2. 对偶空间

对于有限维矢量空间![[公式]](https://www.zhihu.com/equation?tex=V)而言，所有线性映射![[公式]](https://www.zhihu.com/equation?tex=V%5Cto%5Cmathbb%7BR%7D)构成一个矢量空间![[公式]](https://www.zhihu.com/equation?tex=V%5E%2A)  ![[公式]](https://www.zhihu.com/equation?tex=V)与![[公式]](https://www.zhihu.com/equation?tex=V%5E%2A)的一一对应 

(只要构造 线性映射空间的几个基映射 即可)

#### 3. word2vec

实现了 word2vec 模型并使用随机梯度训练下降 (SGD)训练词向量。

代码详见 https://github.com/wangxidong06/CS224N/tree/master/a2





## Lecture 5 dep-parsing

### Slides:

#### 1. 一些基本知识

### Suggested Readings:





## Lecture 6 rnn-lm

### Slides:

#### 1. 一些基本知识

### Suggested Readings:



## Assignment 3

#### 1 pytorch

PyTorch Tutorial Session[[colab notebook](https://colab.research.google.com/drive/1Z6K6nwbb69XfuInMx7igAp-NNVj_2xc3?usp=sharing)] [[preview](http://web.stanford.edu/class/cs224n/materials/CS224N_PyTorch_Tutorial.html)] 

#### 2. Neural Transition-Based Dependency Parsing  

实现了一个基于神经网络的依赖解析器，目标是最大限度地提高 UAS（未标记附件分数）指标的性能。

代码详见https://github.com/wangxidong06/CS224N/tree/master/a3



## Lecture 7 fancy-rnn

### Slides:

#### 1. 一些基本知识

### Suggested Reading:













