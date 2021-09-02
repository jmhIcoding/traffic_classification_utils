# website_fingerprinting

目前本项目支持如下模型：


- Deep Fingerprinting 

- SDAE

- LSTM

- CNN


剩余两个是统计机器学习模型：【 目前这两个模型没有适配好，但是里面的特征提取是有效的】

- CUMUL

- AppScanner


# 使用方法

## 数据准备

首先，需要准备好数据格式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013165821472.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
需要将网络流量整理为如上的6个文件，并放在同一个目录，文件名如上。

```python
X_train_pkt_length.pkl : 包长序列，训练集。
X_valid_pkt_length.pkl : 包长序列，验证集。
X_test_pkt_length.pkl : 包长序列，测试集。
y_train_pkt_length.pkl : 流量标签，训练集。
y_valid_pkt_length.pkl : 流量标签，验证集。
y_test_pkt_length.pkl : 流量标签，测试集。
```
其中，`X_*_pkt_length.pkl` 是一个使用pickle.save()保存的numpy 矩阵，它的形状为 $m\times l$ 。其中m是样本个数，l是包长序列的长度，**同一数据集所有样本的包长序列需要填充到相同的长度** 。
`y_*_pkt_length.pkl` 也是一个pickle.save()保存的numpy矩阵，它的形状为 $m\times1$，m表示样本个数，第i个元素都是整数，表示对应的训练集、验证集、测试集第i个样本的标签。
数据集的保存需要使用类似如下的步骤：

```python
        with gzip.GzipFile(path_dir+"/"+"X_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_test,fp,-1)

        with gzip.GzipFile(path_dir+"/"+"y_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_test,fp,-1)
```

**训练集的包长序列的样本数目需要等于训练集的流量标签序列的样本数。
验证集的包长序列的样本数目需要等于验证集的流量标签序列的样本数。
测试集的包长序列的样本数目需要等于测试集的流量标签序列的样本数。**

项目提供了一个示例数据集 app_dataset，它是一个55分类的数据集，每条样本的包长序列长度为1000,不足的填充0，超过1000的就截断。


---
## 修改数据目录
在按照上述步骤准备好数据后，需要修改数据目录。
修改`website_fingerprinting/data_utils.py`  文件里面的`NB_CLASSES` 变量 和 默认数目集目录`dataset_dir` 变量。
其中`NB_CLASSES` 变量是数据集不同标签的数目。
`dataset_dir` 是默认数据集的目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013171642726.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

---
## 配置模型
在运行模型之前，需要先修改他们的配置文件。
目前，各个模型的配置文件以模型名命名的目录下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013171123714.png#pic_center)
例如，对于Deep fingerprinting模型，它的配置文件为df目录下的df_model_config.py。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013171223270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
修改模型文件：**修改里面的类别数目和包长序列的长度参数。** 对于里面需要修改的参数，各模式文件都做了标注。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013171257550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
## 运行模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013172217604.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
运行 `X_example.py ` 进行模型的训练，其中X可以是df,cnn,lstm,sdae。
运行`X_eval.py` 进行模型的测试，其中X可以是df,cnn,lstm,sdae.

例如：
在自带的app_dataset数据集运行的 `df_example.py` 的结果为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013173850869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

