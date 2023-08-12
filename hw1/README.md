# Homework 1复盘

## forward

- 这里主要是熟悉 numpy 的各种 api。

## backward

- 需要注意的是`backward`的输入是两个 Tensor，返回的也是 List[Tensor]。因此其实要利用 op 来进行计算
- 拓扑排序是经典算法以及实现
- 关于`optimizer.step()`，我们不能简单的用 Tensor 去做减法，这样会造成计算图的混乱；我们必须创建一个新的 `Tensor`。

## 其他

- 我试图全局使用 `lazy_mode` ，但是注意到，在一旦使用`Tensor.shape`，就会触发计算。我试图思考如何在不进行计算的前提下，获得每个 tensor 的 shape 以及整体运行逻辑。