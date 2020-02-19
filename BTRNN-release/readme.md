# BT_RNN

运行步骤：
1. 环境依赖请参考docker/requirements.txt文件，系统环境依赖请参考docker/Dockerfile文件
2. 下载UCF11数据集并解压：https://www.crcv.ucf.edu/data/UCF11_updated_mpg.rar
3. 数据预处理，`python3 dataset/UCF11.py`查看参数并作数据预处理
4. 修改数据读取路径和结果保存路径，见：experiment_UCF11.py文件18-20行
5. 执行：`python3 experiment_UCF11.py` 查看执行参数，如：`python3 experiment_UCF11.py btlstm 2 4 2`


Ref:

1. De Lathauwer, Lieven. "Decompositions of a higher-order tensor in block terms—Part II: Definitions and uniqueness." SIAM Journal on Matrix Analysis and Applications 30.3 (2008): 1033-1066.
2. Novikov, Alexander, et al. "Tensorizing neural networks." Advances in Neural Information Processing Systems. 2015.
3.  Yinchong Yang, Denis Krompass, Volker Tresp. "Tensor-Train Recurrent Neural Networks for Video Classification." In International Conference on Machine Learning 34 (ICML-2017).
