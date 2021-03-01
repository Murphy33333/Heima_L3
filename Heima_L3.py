import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram,ward
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
#数据加载
data=pd.read_csv('D:\黑马课程\第三课\TClass\car_data.csv',encoding='gbk')
train_x=data[['人均GDP','城镇人口比重','交通工具消费价格指数','百户拥有汽车量']]

#数据规范化到【0，1】
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
print(train_x)

#使用kmeans进行聚类分为四类
kmeans=KMeans(n_clusters=4)
kmeans.fit(train_x)
train_y=kmeans.predict(train_x)
print(train_y)
result=pd.concat((data,pd.DataFrame(train_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(result)

for i in range(0,4):
    city_result=result[result['聚类结果']==i]
    print('----------------------------------------------split--------------------L3------------------------------------')
    print("城市划分分类"+str(i))
    print(city_result)
# # 使用kmeans进行层次
# model=AgglomerativeClustering(linkage='ward',n_clusters=4)
# # #fit+predit
# y=model.fit_predict(train_x)
# print(y)
# linkage_matrix=ward(train_x)
# dendrogram(linkage_matrix)
# plt.show()
#

#层次聚集