# ChineseNER
本项目使用
python3.6 + pytorch


对命名实体识别不了解的可以先看一下<a href="https://mp.weixin.qq.com/s/DUxbiVUykVSCLeYxRYd-LA" target="_blank">这篇文章</a>。顺便求star～

这是最简单的一个命名实体识别BiLSTM+CRF模型。

## 数据
data文件夹中有三个开源数据集可供使用，玻森数据 (https://bosonnlp.com) 、1998年人民日报标注数据、MSRA微软亚洲研究院开源数据。其中boson数据集有6种实体类型，人民日报语料和MSRA一般只提取人名、地名、组织名三种实体类型。

当前使用的训练数据，132篇pdf数据。

## 相关文件夹与脚本介绍
create_data: 保存创建训练所使用数据
    key_dict.txt: 相关key字典
    deal_key_dict.txt: 细处理后的key字典
    deal_dict.py: 处理相关字典
    pdf2text.py: 对pdf解析后的doc.json进行处理，输出text文本
    create_data.py: 对生成对text文本进行细处理，输出用于训练的文本
model: 存储训练出来的模型文件
pkl_model: 存储pkl文件
BiLSTM_CRF.py: 训练模型代码
train.py: 训练预测入口函数
resultCal.py: 工具函数




## 准确率
在测试下： 参数并没有调的太仔细，boson数据集的f值在70%~75%左右，人民日报和MSRA数据集的f值在85%~90%左右。（毕竟boson有6种实体类型，另外两个只有3种）

当前模型的验证集准确率下：
1. zhun: 0.9921856500118399
2. zhao: 0.9886739027843322 
3. f: 0.9904266635149509



