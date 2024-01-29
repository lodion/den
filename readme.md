目前是复现《基于用户窗口的内部威胁检测机制》这篇论文，大致内容是用户身份存在被盗可能，攻击者行为在window系统中直接体现为与窗口交互，所以通过表征窗体行为，基于高斯模型和欧氏距离进行不同用户间身份的区别。


代码部分是建立了userdata类，基于这个类，随机生成表征窗体行为的数据，并通过算法进行区别。


复现论文的路径是UEBA\特征工程\基于用户窗体行为的异常检测。
