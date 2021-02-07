#画图
import matplotlib.pyplot as plt
x=[0,0,0.062,0.062,0.085,0.085,0.224,0.224,0.261,0.261,0.356,0.356,0.511,0.511,0.66,0.66,0.816,0.816,0.94,0.94,1]
y=[0,0.231,0.231,0.377,0.377,0.588,0.588,0.667,0.667,0.764,0.764,0.8,0.8,0.849,0.849,0.921,0.921,0.96,0.96,1,1]
x=[i*80 for i in x]
y=[i*80 for i in y]

img = plt.imread("/Users/ren/Downloads/2.jpeg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-5, 80, -5, 80])
plt.plot(x, y, 'bo-',label="ROC", linewidth=1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.legend(loc='best',edgecolor='red') #设置图例边框颜色
plt.legend(loc='best',facecolor='red') #设置图例背景颜色,若无边框,参数无效
plt.show()