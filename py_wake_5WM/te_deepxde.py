import numpy as np
import pandas as pd
import deepxde as dde
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(csv_path):
    """数据准备函数"""
    df = pd.read_csv(csv_path)
    X = df['WindSpeed'].values.reshape(-1, 1)
    y = np.hstack([df[['Power']].values, df[['Ct']].values * 10000])

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 保存数据
    np.savetxt("dataset.train", np.hstack((X_train, y_train)))
    np.savetxt("dataset.test", np.hstack((X_test, y_test)))
    return X_train, y_train


def build_model():
    """构建简化版神经网络模型"""
    data = dde.data.DataSet(
        fname_train="dataset.train",
        fname_test="dataset.test",
        col_x=(0,),
        col_y=(1, 2),
        standardize=True
    )

    # 简化网络结构：1输入 -> 30神经元 -> 20神经元 -> 2输出
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(
        [1, 50, 50, 2],activation=activation,kernel_initializer=initializer)

    model = dde.Model(data, net)
    model.compile(
        optimizer="adam",
        lr=0.05,
        metrics=["l2 relative error"]
    )
    return model


if __name__ == "__main__":
    # 准备数据
    prepare_data("wind_turbine_results.csv")

    # 训练模型
    model = build_model()
    losshistory, train_state = model.train(iterations=50000)

    # 保存训练损失曲线（但不自动绘图）
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # 测试预测
    test_data = np.array([[3.0], [5.0], [7.0]])
    predictions = model.predict(test_data)

    print("\n测试预测:")
    for i in range(len(test_data)):
        print(f"风速 {test_data[i][0]:.1f}m/s -> 功率 {predictions[i][0]:.2f}kW, Ct {predictions[i][1]/10000:.3f}")

