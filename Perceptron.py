import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
###--------------------------------------------------------------------------###
###・インスタンス生成時の初期化
###・教師データを保存
###--------------------------------------------------------------------------###
    def __init__(self, correctSet):
        self.correctSet = correctSet

###--------------------------------------------------------------------------###
###・テスト前に行う処理
###・パラメータをセットする
###・拡張特徴ベクトル
###--------------------------------------------------------------------------###
    def setParameter(self, w):
        self.w = w
###--------------------------------------------------------------------------###
###・各ニューロンの発火時の処理（重み付け和）
###・モデルの出力値を訓練標本ごとにリストに保存
###・
###--------------------------------------------------------------------------###
    def getClasslabel(self, xi):
        #重み付け和計算
        tmp = np.sum(self.w * xi)

        #分類
        if tmp <= 0:
            return 0
        else:
            return 1

    def getClasslabelVec(self, trainingSet):
        Y = []
        for xi in range(len(trainingSet)):
            y = self.getClasslabel(trainingSet[xi])
            Y.append(y)
        return Y
###--------------------------------------------------------------------------###
###・テスト
###・教師データとモデル出力値の比較
###・
###--------------------------------------------------------------------------###
    def Test(self, trainingSet):
        #モデル出力値
        y = self.getClasslabelVec(trainingSet)
        for xi in range(len(y)):
            #教師データとモデル出力値の比較
            if self.correctSet[xi] != y[xi]:
                print("パラメータを再設定してください！")
            else:
                print(str(trainingSet[xi]) + " -> " + str(y[xi]))
###--------------------------------------------------------------------------###
###・描画処理、matplotlib
###・訓練セットと正解セットを基に描画
###・連続曲線（分類境界）を引くための準備
###--------------------------------------------------------------------------###
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #訓練データ
        for xi in range(len(trainingSet)):
            if self.correctSet[xi] == 1:
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
            w0x0 = self.w[0]*XS[i][0]
            w1x1 = self.w[1]*XS[i][1]
            f_real.append((-w0x0-w1x1)/self.w[-1])
        ax.scatter(xs, f_real, c='green')

        ax.set_title('Original Sample Space')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        #plt.savefig('result.png')
        plt.show()

###--------------------------------------------------------------------------###
###・インスタンス生成
###--------------------------------------------------------------------------###
##############################訓練セット : X = (x0, x1, x2)######################
trainingSet = np.array([(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)])

correctSet_AND = np.array([0, 0, 0, 1])
correctSet_OR = np.array([0, 1, 1, 1])
correctSet_NAND = np.array([1, 1, 1, 0])
correctSet_XOR = np.array([0, 1, 1, 0])

##############################インスタンス生成#####################################
classifier_AND = Perceptron(correctSet_AND)
classifier_OR = Perceptron(correctSet_OR)
classifier_NAND = Perceptron(correctSet_NAND)
classifier_XOR = Perceptron(correctSet_XOR)

##############################パラメータセット：W = (w0, w1, w2)###################
parameterSet_AND = np.array([-1.5, 1.0, 1.0]) #np.array([-0.7, 0.5, 0.5])
parameterSet_OR = np.array([-0.5, 1.0, 1.0]) #np.array([-0.2, 0.5, 0.5])
parameterSet_NAND = np.array([1.5, -1.0, -1.0]) #np.array([0.7, -0.5, -0.5])

classifier_AND.setParameter(parameterSet_AND)
classifier_OR.setParameter(parameterSet_OR)
classifier_NAND.setParameter(parameterSet_NAND)
classifier_XOR.setParameter(parameterSet_AND)
##############################テスト(単層パーセプトロン)###########################
#classifier_AND.Test(trainingSet)
#classifier_OR.Test(trainingSet)
#classifier_NAND.Test(trainingSet)
##############################曲線描画(単層パーセプトロン)##########################
#classifier_AND.plot()
#classifier_OR.plot()
#classifier_NAND.plot()
##############################テスト（多層パーセプトロン）###########################
s1 = classifier_NAND.getClasslabelVec(trainingSet)
s2 = classifier_OR.getClasslabelVec(trainingSet)

S = np.ones((4,3))
for i in range(len(S)):
    S[i][1] = s1[i]
    S[i][2] = s2[i]

classifier_XOR.Test(S)
