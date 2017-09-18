import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    #def __init__(self):

    #パラメータをセットする
    def setParameter(self, w):
        self.w = w

    #単層パーセプトロン
    def singlePerseptron(self, xi):
        #重み付け和計算
        tmp = np.sum(self.w * xi)

        #分類
        if tmp <= 0:
            return 0
        else:
            return 1

    #多層パーセプトロン
    def MultPerceptron(self, xi):
        s1 = self.singlePerseptron(xi)
        s2 = self.singlePerseptron(xi)
        y = self.singlePerseptron(xi)
        return y

##############################インスタンス生成#####################################
classifier = Perceptron()

##############################訓練セット : X = (x0, x1, x2)####################
trainingSet = np.array([(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)])
correctSet = np.array([0, 0, 0, 1])

##############################パラメータセット：W = (w0, w1, w2)###################
parameterSet = np.array([-1.5, 1.0, 1.0])
classifier.setParameter(parameterSet)

##############################テスト#############################################
for xi in range(len(trainingSet)):
    y = classifier.singlePerseptron(trainingSet[xi])

    if correctSet[xi] != y:
        print("パラメータを再設定してください！")
    else:
        print(str(trainingSet[xi]) + " -> " + str(y))
##############################曲線描画###########################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#訓練データ
for xi in range(len(trainingSet)):
    if correctSet[xi] == 1:
        ax.scatter(trainingSet[xi][1],trainingSet[xi][2], c='red')
    else:
        ax.scatter(trainingSet[xi][1],trainingSet[xi][2], c='blue')

#連続関数のプロット用ベクトル
xs = np.linspace(-0.25, 1.25, 500)
xs0 = np.ones((500, 1))
XS = []
for i in range(len(xs0)):
     XS.append(np.append(xs0[i], xs[i]))
#連続関数
f_real = []
for i in range(len(XS)):
    w0x0 = parameterSet[0]*XS[i][0]
    w1x1 = parameterSet[1]*XS[i][1]
    f_real.append((-w0x0-w1x1)/parameterSet[-1])
ax.scatter(xs, f_real, c='green')

ax.set_title('Original Sample Space')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
#plt.savefig('result.png')
plt.show()
