import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow


# 2. 导入数据集
data = pd.read_csv('A:\桌面\Income1.csv', index_col=0)
print(data)


# 分析数据集，可以根据实际情况进行描述，比如查看数据的形状、数据类型、是否有缺失值等
print(f"数据集形状：{data.shape}")
print(f"数据类型：{data.dtypes}")
data['Education'] = pd.to_numeric(data['Education'], errors='coerce')
data['Income'] = pd.to_numeric(data['Income'], errors='coerce')

# 3. 重新读入数据并设置第一列为索引
print(data.head())
# "https://blog.csdn.net/weixin_40992494/article/details/104535719"
# 4. 画散点图
plt.scatter(data['Education'], data['Income'])
plt.xlabel('Education')
plt.ylabel('Income')
plt.show()

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# Education = scaler.fit_transform(data['Education'].values.reshape(-1, 1))
# Income = scaler.fit_transform(data['Income'].values.reshape(-1, 1))
# 数据归一化

# 5. 数据预处理
Education = data['Education'].values.reshape(-1, 1)
Income = data['Income'].values

# 6. 建立神经网络模型
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(1, input_dim=1, activation='linear'))
model.summary()
# 分析参数个数：输入层到输出层有 1 个权重(这里是斜率)和 1 个偏置(截距)，共 2 个参数。

# 7. 模型编译
model.compile(optimizer='adam', loss='mse')
# optimizer（优化器）是用于更新模型参数以最小化损失函数的算法。
# loss（损失函数）用于衡量模型预测值与真实值之间的差异。

# 8. 模型训练
model_complete = model.fit(Education, Income, epochs=5000, verbose=1)

# 9. 模型预测
predicted_income = model.predict(np.array([10, 15]).reshape(-1, 1))
print(f"预测的收入：{predicted_income}")
# 对比原数据检查预测准确性，需结合具体原数据进行分析。
