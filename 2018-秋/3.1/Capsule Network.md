# Capsule Network

## CNN 存在的挑战

在深度学习中，CNN擅长抽取 **feature**，但是对于各个 **feature** 之间的空间关系解释的不够高效。例如下图中的图像，简单的CNN网络会认为它是一个人脸素描。
<center>![Alt Text](./img/face.png)</center>

一个简单的CNN可以正确地抽取诸如 **嘴**，**鼻子**，**眼睛** 这样的相关信息，但是它不能识别这些特征的空间信息，于导致针对于脸的识别率过高。![Alt Text](./img/face-cnn.png)

现在，我们做如下考虑，我们不仅考虑 **feature** 的相似度，我们还考虑 **feature** 的空间信息，比如大小和方向。我们把之前每个神经元的输出认作这个 **feature** 的可能性，我们现在把每个神经元当做一个 **向量**，表示为 **(likehood, orientation, size)**。在这些空间信息的参与下，我们会发现这个图像的 眼睛，鼻子，嘴巴的空间信息并不是很合适，于是它判断为人脸的可能性也会随之降低。![Alt Text](./img/face-capsule.png)

## Equivariance(同变性)
理论上来讲，Capsule Network 相对于 CNN 会更为简单，因为 相对于具有 **Equivariance** 的两个 **feature** CNN不能直接识别为相同的 **feature**，而Capsule Network识别出来它们只是简单地做了空间变换，如下图所示：![Alt Text](./img/equivariance.png)
我的理解是这样的，作为一个 CNN 网络就像下图所示，它做识别的时候，不会直接认为做了旋转的脸是脸，至少要通过其它步骤才能识别出来它是脸旋转得来的。而 Capsule Network 会直接认为它是带有旋转的脸。所以

## Capsule
Capsule 可以看做是神经网络的一个集合，它不仅包含**相似性信息(likehood)**，还包括其它指定的特征。如下图所示：![Alt Text](./img/number.png)
在第一行中，我们使用一个单一的神经元来预测它是 **7** 的可能性，在第二三行中我们添加了一个属性 **方向**，我们可以采用某种策略来进行判定，比如我们认为第一个数字的向量为 $\mathbf{v} = (0, 9) \text{并按照如下方式进行判断：} \parallel v \parallel = \sqrt{0^2 + 0.9^2}=0.9$。我们还可以向着capsule中添加其它的 神经元 来丰富这个capsule。

## Dynamic routing
Dynamic routing 组合 capsule 产生 父capsule，并且它计算每个 capsule 的输出。
下面从直觉的角度来讲一下 **Dynamic routing** 的过程。如下图所示，存在三个脸部素描：![Alt Text](./img/faces.png)
我们测试每个素描眼睛和嘴巴的水平宽度分别为 $$s^{(1)}=(100,66)~~~~s^{(2)}=(200,131)~~~~s^{(3)}=(50,33)$$ 我们假定 $W_m=2, W_e=3$ 然后我们通过 capsule 计算一个 **vote** 对于 $s^{(1)}$ 而言：$$v_m^{(1)} = W_m \times width_m = 2 \times 200 = 200 \\
v_e^{(1)} = W_e \times width_e = 3 \times 66 = 198$$ 我们发现 $v_e^{(1)}$ 和 $v_m^{(1)}$ 非常相近，这个特点在其它素描上也有体现。根据经验我们知道嘴巴和眼睛的宽度比例就是 $3:2$， 所以我们可以识别父capsule为一个脸部素描。当然，我们可以通过添加其它的一些属性来使得判断更为准确。

在 **Dynamic routing** 的过程中我们将一个输入 capsule 通过一个 转换矩阵（transformation matrix）W 形成一个 vote，并且将具有相似投票的capsule分组。这些投票最终形成父capsule的输出向量。 

## Calculating a capsule out
对于一个capsule而言，输入Capsule $u_i$ 和 输出Capsule $v_j$ 是矩阵，如下图所示:![Alt Text](./img/capsule-network.png)
我们采用 $\mathbf{transformation}~\mathbf{matrix}~\mathbf{W}_{ij} $ 计算前一层的 capsule $\mathbf{u}_i$。举例来说，我们使用一个 $p\times k$ 矩阵，将 $u_i$ 转换为 $\widehat{u}_{j\mid i}$($(p\times k) \times (k\times 1) \Rightarrow p\times 1$)。接下来我们通过 **weghts** $c_{ij}$ 来计算一个 **weighted sum** $s_j$ $$\widehat{u}_{j\mid i} = W_{ij}u_i \\
s_j=\sum_ic_{ij}\widehat{u}_{j\mid i}$$

$c_{ij}$是**耦合系数(coupling coefficients)**，它通过 **iterative dynamic process** 进行计算（下面会讨论到）。从理论上来说，$c_{ij}$用于测量 capsule i 激活 capsule j 的可能性。

我们设计了一个squashing 函数，用以替代激活函数，它作用于 $s_j$，以用来保证 $v_j$中的项在 0 和 1 之间。这个函数将小向量缩水为0向量，大向量变成单位向量。$$v_j = \frac{\parallel s_j \parallel^2}{1 + \parallel s_j \parallel^2} \frac{s_j}{\parallel s_j \parallel} \\
v_j \approx \parallel s_j \parallel s_j \qquad 如果 s_j 很小 \\
v_j \approx \frac{s_j}{\parallel s_j \parallel} \qquad 如果s_j 很大$$

## iterative dynamic process
在 capsule 中，我们使用 **iterative dynamic process** 来计算 capsule 的的输出，这里我们计算一个中间值 c_{ij}，也就是我们前文提到的 **耦合系数(coupling coefficient)**。![Alt Text](./img/iterative_dynamic_routing.png)

回顾我们之前的计算:$$\widehat{u}_{j\mid i} = W_{ij}u_i \\
s_j=\sum_ic_{ij}\widehat{u}_{j\mid i} \\
v_j = \frac{\parallel s_j \parallel^2}{1 + \parallel s_j \parallel^2} \frac{s_j}{\parallel s_j \parallel}$$

从直觉上来说，$\widehat{u}_{j \mid i}$ 是 capsule i 在capsule j 的输出上的 **preduction（vote）**。如果激活向量（activity vector）和 这个 preduction 十分接近，那么我们可以说这二者是十分相关的。这个相似性是通过 **preduction** 和 **activity vector**的内积来测量的：$$b_{ij} \leftarrow \widehat{u}_{j \mid i} \cdotp v_j$$

这个相似性分数（similarity score）$b_{ij}$ 将**相似性**和其它**特征属性**都考虑在内了。同样，如果 $u_i$ 比较小，那么 $b_{ij}$ 也会比较小，因为 $b_{ij}$ 依赖于 $\widehat{u}_{j \mid i}$，而 $\widehat{u}_{j \mid i}$ 正比于 $u_i$。

耦合系数(coupling coeffients) $c_{ij}$按照如下方式计算:$$c_{ij} = \frac{\exp b_{ij}}{\sum_k \exp b_{iK}}$$

为了使 $b_{ij}$ 更准确一点我们多次迭代更新（通常采用三次迭代）。$$b_{ij} \leftarrow b_{ij} + \widehat{u}_{j \mid i} \cdotp v_j$$

下面是 **Dynamic routing** 最终的伪代码

1. **procedure** ROUTING($\widehat{u}_{j \mid i}, r, l$)
2. $\qquad$ for all capsule $i$ in layer l and capsule $j$ in layer $(l + 1)$: $b_{ij} \leftarrow 0$.
3. $\qquad$ **for** $r$ iterations **do**
4. $\qquad$$\qquad$ for all capsule $i$ in layer $l$: $\mathbf{c_i} \leftarrow \mathbf{softmax}(\mathbf{b_i})$
5. $\qquad$$\qquad$ for all capsule $j$ in layer $(l + 1)$: $\mathbf{s_j} \leftarrow \sum_ic_{ij} \widehat{\mathbf{u}}_{j \mid i}$
6. $\qquad$$\qquad$ for all capsule $j$ in layer $(l + 1)$: $\mathbf{v}_j \leftarrow \mathbf{squash}(\mathbf{s_j})$
7. $\qquad$$\qquad$ for all capsule $i$ in layer $i$ and capsule $j$ in layer $(l + 1)$: $b_{ij} \leftarrow b_{ij} + \widehat{\mathbf{u}}_{j \mid i}\cdotp v_j$
8. $\qquad$ **return** $v_j$

## Max pooling 的缺点
max pooling 只会保留前一层网络的最大 feature，而丢弃其它的的 feature。Capsule 保留了前一层网络的 feature 的 weighted sum。所以对于识别叠加的特征，Capsule 比 Max pooling 更合适。

## CapsNet architecture
下面我们来描述一下这个网络架构:![Alt Text](./img/capsule-architecture.png)

我们使用这个网络来进行 MNIST 数据集上的的手写数字识别。  

1. 首先图片喂给一个标准的 ReLU Conv层。它使用 $256 \times 9 \times 9$ 卷积核产生了一个 $256 \times 20 \times 20$的输出。  
2. 之后 $1.$ 中的输出喂给 PrimaryCapsule 他是一个修改过的支持capsule的卷积层。它产生一个 8-D 的 向量来替代一个值。PrimaryCapsule 使用 $8\times 32$ 的卷积核来产生 32 8-D capsules（可以使用8个神经元组合在一起来产生一个capsule），PrimaryCapsule 使用 $9 \times 9$ 的卷积核并且 $strid=2$来进行降维，最终产生 $32 \times 6 \times 6$ 个capsule（$20 \times 20 \Rightarrow 6 \times 6~~~~\lfloor\frac{20-9}{2}\rfloor + 1 = 6$）。  
3. 之后我们进入 DigiCaps 这里我们使用  $16 \times 8$ 转换矩阵 $\mathbf{W_{ij}}$ 将 8-D capsule 转化为 10-D capsule。  
4. 之后我们使用激活函数为 $\mathbf{ReLU}$ 的 fully connected layer来进行分类的最后一步，具体的网络如下所示。![Alt Text](./img/capsule-layers.png)

## Loss Function
代价函数采用margin loss（这个我没有看过，也不知道为啥用这个，可能之后会补充）。对于每个分类有如下公式：$$L_c=T_c\max(0, m^+ -\parallel v_c \parallel)^2 + \lambda(1-T_c)\max(0, \parallel v_c \parallel - m^-)^2$$
其中如果c类被正确识别 $T_c=1$。其他参数采用如下数据$$m^+=0.9\\
m^-=0.1\\
\lambda=0.5$$总损失就是单个类别的损失之和。

## 参考资料 
* [Dynamic Routing Between Capsules Blog](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/) 本文主要参考了这一篇博客，所有图片都来自与这篇博客，针对他所说的话有的加上了自己的理解，有的只是单纯的翻译。
* [Dynamic Routing Between Capsules 论文](https://arxiv.org/pdf/1710.09829.pdf)

### 如果有幸，您看完了这篇文章，欢迎批评指正，大家一起进步。
