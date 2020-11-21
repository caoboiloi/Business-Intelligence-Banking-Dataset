import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

def readDataset():
	# Đọc dataset
	data = pd.read_csv("BankAdditional/bank-full.csv")
	data = data.dropna()

	# Row: 41188 record - col: 21 fields
	# print(data.shape)

	# Tiêu đề
	# print(list(data.columns))

	# In 5 dòng đầu
	# print(data.head())

	# Dữ liệu cột education
	# print("Dự liệu ban đầu cột education:",data['education'].unique())

	# Nhóm dữ liệu basic.4y - basic.6y - basic.9y thành basic

	data['education'] = np.where(data['education'] == 'basic.4y','basic',data['education'])
	data['education'] = np.where(data['education'] == 'basic.6y','basic',data['education'])
	data['education'] = np.where(data['education'] == 'basic.9y','basic',data['education'])

	# Kiểm tra:
	# print('---------------------------------')
	# print("Dự liệu sau khi gom nhóm cột education:",data['education'].unique())
	X = data.loc[:, data.columns != 'y']
	# print('Các biến độc lập: ')
	# print(X.head())
	# print(X.columns)
	# print('Số lượng biến độc lập: ',len(X.columns))
	return data

def numberVariable(data):
	# Chuyển đổi biến chuỗi thành các cột chứa dữ liệu dạng binary (có hoặc không)
	# vd: marital : married -> marital_married : 1 nếu khách hàng đó có kết hôn hay marital_married : 0 nếu khách hàng ko có kết hôn
	vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
	for var in vars:
	    cat_list = 'var' + '_' + var
	    cat_list = pd.get_dummies(data[var], prefix = var)
	    data1 = data.join(cat_list)
	    data = data1
	vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
	data_vars = data.columns.values.tolist()
	to_keep = [i for i in data_vars if i not in vars]
	data_final = data[to_keep]

	X = data_final.loc[:, data_final.columns != 'y']
	# Test title dataset
	# print('Các biến độc lập: ')
	# print(X.head())
	# print(X.columns)
	# print('Số lượng biến độc lập: ',len(X.columns))
	# print(data_final.columns.values)
	# print(data_final.head(5))

	return data_final


def Experimental(data_final):
	data_job_true_1 = 0
	data_job_false_1 = 0
	data_job_true_0 = 0
	data_job_false_0 = 0
	for i in range(len(data_final)):
		if data_final['y'][i] == 1 and data_final['job_blue-collar'][i] == 1:
			data_job_true_1 += 1
		elif data_final['y'][i] == 1 and data_final['job_blue-collar'][i] == 0:
			data_job_false_1 += 1
		elif data_final['y'][i] == 0 and data_final['job_blue-collar'][i] == 1:
			data_job_true_0 += 1
		elif data_final['y'][i] == 0 and data_final['job_blue-collar'][i] == 0:
			data_job_false_0 += 1

	print('======================= THỰC NGHIỆM =======================')
	print('Khách hàng đăng ký',len(data_final[data_final['y'] == 1]))
	print('Khách hàng không đăng ký',len(data_final[data_final['y'] == 0]))

	print('Khách hàng đăng ký có nghề nghiệp là blue-collar: ',data_job_true_1)
	print('Khách hàng đăng ký có nghề nghiệp là nghề khác: ',data_job_false_1)
	print('Khách hàng không đăng ký có nghề nghiệp là blue-collar: ',data_job_true_0)
	print('Khách hàng không đăng ký có nghề nghiệp là nghề khác: ',data_job_false_0)
	# print(data_final['job_blue-collar'])
	# print(data_final['education_basic'])


# Trực quan hóa dữ liệu:
def visualizationChart(data):

	# Đồ thị cột biểu hiện biến phân loại y: y

	# print(data['y'].value_counts())

	# Đồ thị cột
	# 1 là true, 0 là false
	sns.countplot(x = 'y',data = data, palette ='hls').set_title('Số lượng khách hàng đăng ký một khoản tiền gửi có kỳ hạn')
	plt.tight_layout()
	plt.savefig('count_plot')

	# Số lượng đăng ký cho từng loại nghề nghiệp trong tập dataset: job

	pd.crosstab(data['job'],data['y']).plot(kind='bar')
	plt.title('Số lượng đăng ký của khách hàng theo nghề nghiệp')
	plt.xlabel('Nghề nghiệp')
	plt.ylabel('Số lượng khách hàng')
	plt.tight_layout()
	plt.savefig('frequency_job')

	# Tình trạng hôn nhân: marital
	table=pd.crosstab(data['marital'],data['y'])
	table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
	plt.title('Tỉ suất đăng ký của khách hàng theo tình trạng hôn nhân')
	plt.xlabel('Tình trạng hôn nhân')
	plt.ylabel('Tỉ lệ khách hàng')
	plt.tight_layout()
	plt.savefig('frequency_mariral')

	# Các ngày trong tuần: day_of_week
	pd.crosstab(data['day_of_week'],data['y']).plot(kind='bar')
	plt.title('Số lượng đăng ký của khách hàng theo ngày')
	plt.xlabel('Các ngày trong tuần')
	plt.ylabel('Số lượng khách hàng')
	plt.tight_layout()
	plt.savefig('frequency_dayofweek')

	# giáo dục: education
	table=pd.crosstab(data['education'],data['y'])
	table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
	plt.title('Tỉ suất đăng ký của khách hàng theo tình trạng giáo dục')
	plt.xlabel('Giáo dục')
	plt.ylabel('Tỉ lệ khách hàng')
	plt.tight_layout()
	plt.savefig('frequency_education')

	# các tháng: month
	pd.crosstab(data['month'],data['y']).plot(kind='bar')
	plt.title('Số lượng đăng ký của khách hàng theo tháng')
	plt.xlabel('Tháng')
	plt.ylabel('Số lượng khách hàng')
	plt.tight_layout()
	plt.savefig('frequency_month')

	# tuổi: age
	data.hist('age')
	plt.title('Số lượng đăng ký của khách hàng theo độ tuổi') 
	plt.xlabel('Tuổi') 
	plt.ylabel('Số lượng khách hàng') 
	plt.tight_layout()
	plt.savefig('hist_age')

	# kết quả của chiến dịch tiếp thị trước đó của khách hàng: poutcome

	pd.crosstab (data['poutcome'],data['y']).plot(kind = 'bar') 
	plt.title ('Số lượng đăng ký của khách hàng thông qua tình trạng tiếp thị trước đó') 
	plt.xlabel ('Tình trạng') 
	plt.ylabel ('Số lượng khách hàng')
	plt.tight_layout()
	plt.savefig ('frequency_poutcome')

def subscriptionPercent(data):
	# Tỉ lệ phần trăm những khách hàng đăng ký
	count = len(data['y'])
	count_sub = len(data[data['y']==0])
	count_no_sub = len(data[data['y']==1])

	sub_percent = count_sub/count
	no_sub_percent = count_no_sub/count

	print('tỉ lệ phần trăm khách hàng đăng ký:',round(sub_percent * 100,2),'%')
	print('tỉ lệ phần trăm khách hàng không đăng ký:',round(no_sub_percent * 100,2),'%')

	# Trung bình theo biến phân loại y:
	# print(data.groupby('y').mean())


data = readDataset()
print('=======================================================')
print('------------Percentage------------')
visualizationChart(data)
subscriptionPercent(data)
data_final = numberVariable(data)
Experimental(data_final)