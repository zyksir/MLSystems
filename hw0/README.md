# Homework 0

Public repository and stub/testing code for Homework 0 of 10-714.
由于我未能获得GRADER_KEY,因此注释了所有提交修改的请求。

## Add & Loading MNIST data & softmax_loss

Add 比较简单。 MNIST data 有其特定的格式，很奇怪的是网上搜索搜到的答案都是错的。这里用到东西总结如下：

- `struct.unpack(">4i", f.read(16))`, 这里">"代表大端法，其他具体细节参考[struct format strings](https://docs.python.org/3/library/struct.html#struct-format-strings)。
- 其余的就参考 MNIST data 的格式。大部分数据文件都是一开始几个 int 的 meta info，然后是数据。
- softmax 注意在实际应用的时候，是需要`z - z.max(-1)`来防止溢出的，另外注意 `Z = X@W; Z = Z / Z.sum(-1)` 这个东西在 backward 的时候还会用，因此通常在 forward 的。
- 需要记住的是 
$$
\nabla_{\Theta} \ell_{\text {softmax }}\left(\Theta^T x, y\right)=x\left(z-e_y\right)^T \\
\text{where}\quad z=\frac{\exp \left(\Theta^T x\right)}{1^T \exp \left(\Theta^T x\right)} \equiv \operatorname{normalize}\left(\exp \left(\Theta^T x\right)\right)
$$
- CPP 主要是注意矩阵乘法的实现，以及如何利用矩阵乘法实现剩下的部分。
- 链式法则的推导是很重要的环节，具体应该看课上是怎么推导的。