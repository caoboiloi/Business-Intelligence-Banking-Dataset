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
	# print(data['education'].unique())

	# Nhóm dữ liệu basic.4y - basic.6y - basic.9y thành basic

	data['education'] = np.where(data['education'] == 'basic.4y','basic',data['education'])
	data['education'] = np.where(data['education'] == 'basic.6y','basic',data['education'])
	data['education'] = np.where(data['education'] == 'basic.9y','basic',data['education'])

	# Kiểm tra:
	# print(data['education'].unique())
	return data

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

	print('tỉ lệ phần trăm khách hàng đăng ký:',sub_percent * 100)
	print('tỉ lệ phần trăm khách hàng không đăng ký:',no_sub_percent * 100)

	# Trung bình theo biến phân loại y:
	# print(data.groupby('y').mean())

print('=======================================================')
print('------------Percentage------------')
data = readDataset()
visualizationChart(data)
subscriptionPercent(data)