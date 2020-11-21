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

# Tỉ lệ phần trăm của khách hàng đăng ký và không đăng ký
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
	# Test title dataset
	# print(data_final.columns.values)
	# print(data_final.head(5))
	return data_final

# xử lý dữ liệu: do dataset majority chiếm phần lớn nên ta phải cân bằng minority có lượng dữ liệu như nhau.
# Sau đó train mô hình ở tập dữ liệu mới này.
def createSmote(data_final):
	# Oversampling: 
	# Tạo dữ liệu giả cho tập minority sao cho số phần tử của nó được nhiều lên.
	# Cách đơn giản nhất là lặp lại mỗi điểm trong minority nhiều lần.
	# tránh trường hợp khi phân training test chỉ tách ra majority -> gây sai lệch kết quả kiểm tra độ chính xác
	# ================================================================
	# SMOTE:
	# Với mỗi điểm trong tập minority, tìm k điểm cùng trong minority gần nó nhất rồi dùng tổng có trọng số của các điểm này để tạo ra các điểm dữ liệu mới. 
	
	# Test new dataset
	# print(data_final)
	# Lấy tất cả biến bỏ biến phân loại y
	X = data_final.loc[:, data_final.columns != 'y']
	# Lấy biến phân loại y
	y = data_final.loc[:, data_final.columns == 'y']
	os = SMOTE(random_state=0)
	# Phân data thành 2 phần: training data và testing data -> 30% Test - 80% train
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	# test length Training data, Testing data
	print('Số lượng tập train:',len(X_train))
	print('Số lượng tập test:',len(X_test))
	# title dataset
	feature_rol = X_train.columns

	# dữ liệu được tạo ra hơn 10000 điểm dữ liệu mới
	os_data_X,os_data_y = os.fit_sample(X_train, y_train)

	os_data_X = pd.DataFrame(data = os_data_X,columns = feature_rol)

	os_data_y = pd.DataFrame(data = os_data_y,columns = ['y'])
	# Kiểm tra dữ liệu
	print('length Oversampling:',len(os_data_X))
	print('Số lượng khách hàng không đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==0]))
	print('Số lượng khách hàng đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==1]))
	print('Tỉ lệ khách hàng không đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==0])/len(os_data_X))
	print('Tỉ lệ khách hàng đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==1])/len(os_data_X))
	# Dữ liệu sau khi Oversampling dựa trên dữ liệu tập train data, do đó không có thông tin nào dựa vào tập test data
	# Kết quả dự đoán sẽ được chính xác
	return os_data_X,os_data_y,X_test,y_test

#RFE: Biểu diễn kết quả của quá trình thử và sai các tập con features để đưa ra số lượng features và các features tối ưu cho thuật toán dự đoán.
# Tìm ra số lượng feature tối ưu cho thuật toán đang sử dụng, tránh lặp lại nhiều lần
def recursiveFeatureElimination(data_final,smote_data_X,smote_data_y):
	# title dataset final
	feature_rol = data_final.columns.values.tolist()

	# title dataset final not y
	X = [i for i in feature_rol if i not in 'y']
	
	rfe = RFE(LogisticRegression(), 20)
	rfe = rfe.fit(smote_data_X, smote_data_y.values.ravel())
	# Kiểm tra những đặc tính phù hợp
	print(rfe.support_)
	# print(smote_data_X)
	print(rfe.ranking_)

def logisticRegression(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	model = LogisticRegression(max_iter = 1000)
	model.fit(X_train, y_train)
	return accuracy_score(y_test, model.predict(X_test)),X_test,y_test,X_train,y_train

def probabilityChart(X_test,y_test,X_train,y_train):
	# một công cụ hỗ trợ dánh giá được độ chính xác của thuật toán logistic
	model = LogisticRegression(max_iter = 1000)
	model.fit(X_train, y_train)
	logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Đặc trưng (false positives)')
	plt.ylabel('Độ nhạy (true positives)')
	plt.title('Đường cong ROC')
	plt.legend(loc="lower right")
	plt.savefig('probabilityChart')

print('=======================================================')
print('------------Percentage------------')
data = readDataset()
subscriptionPercent(data)
print('=======================================================')
print('------------Oversampling - SMOTE------------')
# Logistic Regression
data_final = numberVariable(data)
os_data_X,os_data_y,X_test,y_test = createSmote(data_final)

# RFE: tối ưu các feature của tập dataset
# recursiveFeatureElimination(data_final,os_data_X,os_data_y)

# dùng RFE xác định các feature tối ưu cho dataset
# Các feature rank = 1
feature_rol_optimal = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
		'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 
		'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
print('=======================================================')
print('------------recursive Feature Elimination------------')
print('chọn các feature sau khi áp dụng RFE tối ưu:')
print(feature_rol_optimal)
X=os_data_X[feature_rol_optimal]
y=os_data_y['y']
accuracy_score,X_test_final,y_test_final,X_train_final,y_train_final = logisticRegression(X,y)
print('=======================================================')
print('------------Logistic Regression------------')
print('Độ chính xác: ',round(accuracy_score*100,2),'%')
print('=======================================================')
print('------------Probability Chart------------')
probabilityChart(X_test_final,y_test_final,X_train_final,y_train_final)