#coding:utf-8
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import scipy.io as scio
import os
import cv2
import sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as skl_PCA
import tensorflow as tf
path="/Users/ren/Desktop/AR"
train_x=[]
train_y=[]
test_x=[]
test_y=[]
pca = PCA(n_components=35)
for file in os.listdir(path):
    if file[-3:]!="bmp":
            continue
    no = int(file[4:6])#取同一张人脸的编号
    if(no>=3):#取后12张人脸作为训练集
        train_tempImg = cv2.imread(path+'/'+file,0)#读入图片
        train_tempImg = np.array(train_tempImg)
        train_tempImg = train_tempImg.reshape(2000)#将二维图片向量转换为以为特征向量
        train_x.append(train_tempImg)
        train_y.append(int(file[:3]))#取不同人脸编号作为label值
    else:#取前2张人脸作为测试集
        test_tempImg = cv2.imread(path+'/'+file,0)
        test_tempImg = np.array(test_tempImg)
        test_tempImg = test_tempImg.reshape(2000)
        test_x.append(test_tempImg)
        test_y.append(int(file[:3])) 
train_x = np.array(train_x)#将列表形式转换为数组形式
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
train_x.reshape(-1,2000)#确保返回形状为[image_nums,size]
test_x.reshape(-1,2000)
#定义PCA算法               
def PCA(data,r):
    '''
    input
    data:降维数据,维度:[dims,image_nums]
    r:降维维度

    return
    final_data:降维后结果,维度:[image_nums,r]
    data_mean:平均值,维度:[1,image_nums]
    V_r:特征值向量,维度:[dims,r]
    '''
    data=np.float32(np.mat(data))#将输入数据转换为numpy中的array形式
    rows,cols=np.shape(data)#得到行列值
    data_mean=np.mean(data,0)#对列求平均值
    A=data-np.tile(data_mean,(rows,1))#去除平均值
    C=A*A.T #计算得到协方差矩阵
    D,V=np.linalg.eig(C)#求协方差矩阵的特征值和特征向量
    V_r=V[:,0:r]#按列取前r个特征向量
    V_r=A.T*V_r#小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        V_r[:,i]=V_r[:,i]/np.linalg.norm(V_r[:,i])#特征向量归一化  
    final_data=A*V_r#将原数据映射到新的空间上
    return final_data,data_mean,V_r
#重构方法
def re_creat(low_mat,vects,mean_data):
    #re_mat[image_nums,dims]
    re_mat = low_mat * vects.T  + mean_data
    return re_mat

#计算重建误差
def loss_clc(re_mat,img_mat):
    #维度:[dims,image_nums]
    re_mat = np.array(re_mat)
    img_mat = np.array(img_mat)
    t1 = img_mat - re_mat    
    t1 = np.multiply(t1,t1)    
    t2 = np.sum(t1,axis=0)
    t3 = np.mean(t2)#计算分子
    #此处因为254*254会超出numpy.multiply的范围，做了除10的操作，后面求loss时乘了回来
    s1 = img_mat/10
    s1 = np.multiply(s1,s1)   
    s2 = np.sum(s1,axis=0)
    s3 = np.mean(s2)#计算分母
    loss = t3/(s3*100)
    print("重建误差为：%.5f,保留了 %d %%的信息"%(loss,(1-loss)*100))
    return loss
#欧式距离判别器
def ed(num_test,num_train,data_test_new,data_train_new,train_label,test_label):
    true_num = 0#判别正确的值
    for i in range(num_test):#遍历测试集中的图片
        testFace = data_test_new[i,:]#取当前图片的特征值
        diffMat = data_train_new - np.tile(testFace,(num_train,1))
        sqDiffMat = diffMat**2#计算训练数据与测试脸之间的欧式距离
        sqDistances = sqDiffMat.sum(axis=1)#按行求和
        sortedDistIndicies = sqDistances.argsort()#对向量从小到大排序，使用的是索引值,得到一个向量
        indexMin = sortedDistIndicies[0]#距离最近的索引
        if train_label[indexMin] == test_label[i]:#判别是否预测正确
            true_num += 1
    return true_num
#使用PCA降维+欧氏距离判别
def pca_ed(k=200):
    print("当前降维维度k=%d"%(k))
    #train_imgMat,train_label,test_imgMat,test_label = creat_Mat(path,2000)#创建训练集和测试集
    data_train_new,data_mean,V_r=PCA(train_x,k)#用训练集计算特征值，返回训练集降维结果
    num_train = data_train_new.shape[0]#训练脸总数
    num_test = test_x.shape[0]#测试脸总数
    temp_face = test_x - np.tile(data_mean,(num_test,1))#测试集去中心值
    data_test_new = temp_face*V_r #得到测试脸在特征向量下的数据
    data_test_new = np.array(data_test_new) # 转换为数组形式
    data_train_new = np.array(data_train_new)
    true_num = ed(num_test,num_train,data_test_new,data_train_new,train_y,test_y)#计算判别正确的数目
    re_train = re_creat(data_train_new,V_r,data_mean)#降维后重建
    old_mat = train_x.T
    new_mat = re_train.T
    loss = loss_clc(new_mat,old_mat)#计算重构误差
    print("当前测试集大小为：%d,判别正确数量为：%d"%(num_test,true_num))
    print("当前判别准确率为：%.5f"%(true_num/num_test))
    #绘图时的输出
    #print("当前降维维度k=%d,当前判别准确率为：%.5f"%(k,(true_num/num_test)))
    return true_num/num_test

pca_ed()
list_x=[]
list_y=[]
#三个梯队进行测试
for i in range(1,50,2):
    list_x.append(i)
    list_y.append(pca_ed(i))
for i in range(52,400,10):
    list_x.append(i)
    list_y.append(pca_ed(i))
for i in range(500,1000,50):
    list_x.append(i)
    list_y.append(pca_ed(i))
    
plt.plot(list_x, list_y, label="pca_acc", linewidth=1.5)
plt.xlabel('D')#维度
plt.ylabel('acc')#准确率
plt.legend()
plt.show()

