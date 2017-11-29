#Back Propagation

- L 表示神经网络的总层数
- $S_l$表示第$l$层神经网络unit个数，不包括偏差单元`bias unit`
- k表示第几个输出单元
- $\Theta^{(l)}_{i,j}$ 第$l$层到第$l+1$层的权值矩阵的第$i$行第$j$列的分量
- $Z^{(j)}_i$ 第$j$层第$i$个神经元的输入值
- $a^{(j)}_i$第$j$层第$i$个神经元的输出值
- $a^{(j)} = g(Z^{(j)})$

###·chain Rule

$y = g(x)\quad z = h(y)​$

$\Delta x \rightarrow \Delta y \rightarrow \Delta z \quad \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$

$x = g(s) \quad y = h(s) \quad z = k(x,y)$

$\Delta s \rightarrow \Delta x \rightarrow \Delta z \quad \Delta s \rightarrow \Delta y \rightarrow \Delta z$

$\frac{dz}{ds} = \frac{\partial z}{\partial x} \frac{dx}{ds} + \frac{\partial z }{\partial y} \frac{dy}{ds}$

###·loss function

$$
J(\Theta) = -\frac{1}{m}\Bigg[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)} \log(h_\Theta(x^{(i)}))_k + (1- y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k) \Bigg]
$$

我们知道代价函数loss function后，下一步就是按照梯度下降法来计算$\theta$求解loss function的最优		解。使用梯度下降法首先要求出梯度，即偏导项$\frac{\partial}{\partial \Theta^{(l)} _{ij}} J(\Theta)$，计算偏导项的过程我们称为back propagation。

根据上面的feed forward computation 我们已经计算得到了 $a^{(1)}$ ，$a^{(2)}$， $a^{(3)}$ ，$Z^{(2)}$，$Z^{(3)}$。

### ·output layer to hidden layer

$$
\frac{\partial}{\partial \Theta^{(L-1)}_{i,j}}J(\Theta) = \frac {\partial J(\Theta)}{\partial h_\theta(x)_i} \frac{\partial h_\theta(x)_i}{\partial z^{(L)}_i} \frac{\partial z^{(L)}_i}{\partial \Theta^{(L-1)}_{i,j}} = \frac {\partial J(\Theta)}{\partial a^{(L)}_i}\frac{\partial a^{(L)}_i}{\partial z^{(L)}_i} \frac{\partial z^{(L)}_i}{\partial \Theta^{(L-1)}_{i,j}}
$$



$$
h_\Theta(x) = a^{(L)} = g(z^{(L)})
$$

$$
z^{(l)} = \Theta^{(l-1)}a^{(l-1)}
$$

$$
loss(\Theta) =- y^{(i)}\log(h_\Theta(x^{(i)}) ) -(1-y^{(i)})\log(1-h_\Theta(x^{(i)}))
$$

$$
\frac{\partial J(\Theta)}{\partial a^{(L)}_i}=\frac{a^{(L)}_i-y_i}{(1-a^{(L)}_i)a^{(L)}_i}
$$

​		仅仅从激活函数的求导就可以推出
$$
\begin{split}
\frac{\partial g(z)}{\partial z} & = -\left( \frac{1}{1 + e^{-z}} \right)^2\frac{\partial{}}{\partial{z}} \left(1 + e^{-z} \right) \\
\\ & = -\left( \frac{1}{1 + e^{-z}} \right)^2e^{-z}\left(-1\right) \\
\\ & = \left( \frac{1}{1 + e^{-z}} \right) \left( \frac{1}{1 + e^{-z}} \right)\left(e^{-z}\right) \\
\\ & = \left( \frac{1}{1 + e^{-z}} \right) \left( \frac{e^{-z}}{1 + e^{-z}} \right) \\
\\ & = \left( \frac{1}{1+e^{-z}}\right)\left( \frac{1+e^{-z}}{1+e^{-z}}-\frac{1}{1+e^{-z}}\right) \\
\\ & = g(z) \left( 1 - g(z)\right) \\
\\ \end{split}
$$

$$
\frac{\partial a^{(L)}_i}{\partial z^{(L)}_i}=\frac{\partial g(z^{(L)}_i)}{\partial z^{(L)}_i}=g(z^{(L)}_i)(1-g(z^{(L)}_i))=a^{(L)}_i(1-a^{(L)}_i)
$$

$$
\frac{\partial z^{(L)}_i}{\partial \Theta^{(L-1)}_{i,j}}=a^{(L-1)}_{i,j}
$$

​							综上
$$
\begin{split}
\frac{\partial}{\partial \Theta^{(L-1)}_{i,j}}J(\Theta) &=\frac {\partial J(\Theta)}{\partial a^{(L)}_i}\frac{\partial a^{(L)}_i}{\partial z^{(L)}_i} \frac{\partial z^{(L)}_i}{\partial \Theta^{(L-1)}_{i,j}}\\
\\ & = \frac{a^{(L)}_i -y_i}{(1-a^{(L)}_i)a^{(L)}_i} a^{(L)}_i(1- a^{(L)}_i) a^{(L-1)}_j \\
\\ & = (a^{(L)}_i-y_i)a^{(L-1)}_j\\
\\ \end{split}
$$







###·hidden layer to hidden layer

$$
\frac{\partial}{\partial \Theta^{(l-1)}_{i,j}}J(\Theta) =  \frac {\partial J(\Theta)}{\partial a^{(l)}_i}\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i} \frac{\partial z^{(l)}_i}{\partial \Theta^{(l-1)}_{i,j}}(l=2,3,4,...,L-1)
$$

$$
\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i}=\frac{\partial g(z^{(l)}_i)}{\partial z^{(l)}_i}=g(z^{(l)}_i)(1-g(z^{(l)}_i))=a^{(l)}_i(1-a^{(l)}_i)
$$

$$
\frac {\partial z^{(l)}_i}{\partial \Theta^{(l-1)}_{i,j}}=a^{(l-1)}_j
$$

第一项的求解就比较难了，需要用上上面的链式法则
$$
\frac{\partial J(\Theta)}{\partial a^{(l)}_i} =\sum^{S_{l-1}}_{k=1} \bigg[ \frac {\partial J(\Theta)}{\partial a^{(l+1)}_k}\frac{\partial a^{(l+1)}_k}{\partial z^{(l+1)}_k} \frac{\partial z^{(l+1)}_k}{\partial a^{(l)}_{i}} \bigg]
$$

$$
\frac {\partial a^{(l+1)}_k}{\partial z^{(l+1)}_k}=a^{(l+1)}_k (1-a^{(l+1)}_k)
$$

$$
\frac{\partial z^{(l+1)}_k}{\partial a^{(l)}_{i}}=\Theta^{(l)}_{k,i}
$$

$$
\begin{split}
\frac{\partial}{\partial \Theta^{(L-1)}_{i,j}}J(\Theta) &= \sum^{S_{l-1}}_{k=1} \bigg[ \frac {\partial J(\Theta)}{\partial a^{(l+1)}_k}\frac{\partial a^{(l+1)}_k}{\partial z^{(l+1)}_k} \frac{\partial z^{(l+1)}_k}{\partial a^{(l)}_{i}} \bigg]\\
\\&= \sum^{S_{l-1}}_{k=1} \bigg[ \frac {\partial J(\Theta)}{\partial a^{(l+1)}_k}\frac{\partial a^{(l+1)}_k}{\partial z^{(l+1)}_k}\Theta ^{(l)}_i \bigg]\\
\\&=\sum^{S_{l-1}}_{k=1} \bigg[ \frac {\partial J(\Theta)}{\partial a^{(l+1)}_k}a^{(l+1)}_k (1-a^{(l+1)}_k) \Theta^{(l)}_{k,i}\bigg]\\
\end{split}
$$

定义第l层第i个节点的的误差为
$$
\begin{split}\
\\\delta^{(l)}_i &=\frac {\partial}{ \partial z^{(l)}_i}J(\Theta)\\
\\&=\frac{\partial J(\Theta)}{\partial a^{(l)}_i}  \frac{\partial a^{(l)}_i}{\partial z^{(l)}_i}\\
\\&=\frac{\partial J(\Theta)}{\partial a^{(l)}_i}a^{(l)}_i(1-a^{(l)}_i)\\
\\&= \sum^{S_{l-1}}_{k=1} \bigg[ \frac {\partial J(\Theta)}{\partial a^{(l+1)}_k}\frac{\partial a^{(l+1)}_k}{\partial z^{(l+1)}_k}\Theta ^{(l)}_i \bigg]a^{(l)}_i(1-a^{(l)}_i)\\
\\&=\sum^{S_{l-1}}_{k=1} \bigg[ \delta^{(l+1)}_k \Theta^{(l)}_{k,i}\bigg]a^{(l)}_i(1-a^{(l)}_i)\\
\end{split}
$$
输出层的误差为
$$
\begin{split}\
\delta^{(L)}_i&=\frac {\partial J(\Theta)}{ \partial z^{(L)}_i}\\
\\&=\frac{\partial J(\Theta)}{\partial a^{(L)}_i}  \frac{\partial a^{(L)}_i}{\partial z^{(L)}_i}\\
\\ & = \frac{a^{(L)}_i -y_i}{(1-a^{(L)}_i)a^{(L)}_i} a^{(L)}_i(1- a^{(L)}_i) a^{(L-1)}_j \\
\\&= a^{(L)}_i-y_i
\end{split}
$$
final 代价函数的偏导数为
$$
\begin{split}\
\\\frac {\partial}{ \partial z^{(l-1)}_i}J(\Theta)&=\frac {\partial J(\Theta)}{\partial a^{(l)}_i}\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i} \frac{\partial z^{(l)}_i}{\partial \Theta^{(l-1)}_{i,j}}\\
\\&=\frac {\partial J(\Theta)}{\partial z^{(l)}_i}\\
\\&=\delta^{(l)}_i \frac{\partial z^{(l)}_i}{\partial \Theta^{(l-1)}_{i,j}}\\
\\&=\delta^{(l)}_i a^{(l-1)}_i\\
\end{split}
$$
总结

> $$输出层误差 \delta^{L}_i$$
> $$
> \delta^{L}_i=a^{(L)}_i-y_i
> $$
>

> $$隐层误差\delta^{l-1}_i$$
> $$
> \delta^{(l)}_i =\sum^{S_{l-1}}_{k=1} \bigg[ \delta^{(l+1)}_k \Theta^{(l)}_{k,i}\bigg]a^{(l)}_i(1-a^{(l)}_i)
> $$
>

> $$代价函数的偏导项\frac {\partial}{ \partial z^{(l)}_i}J(\Theta)$$
> $$
> \frac {\partial}{ \partial z^{(l-1)}_i}J(\Theta)=\delta^{(l)}_i a^{(l-1)}_i
> $$
> 即
> $$
> \frac {\partial}{ \partial z^{(l)}_i}J(\Theta)=\delta^{(l+1)}_i a^{(l)}_i
> $$
>