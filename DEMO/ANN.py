from joblib.numpy_pickle_utils import xrange
from numpy import *
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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

	data['education'] = where(data['education'] == 'basic.4y','basic',data['education'])
	data['education'] = where(data['education'] == 'basic.6y','basic',data['education'])
	data['education'] = where(data['education'] == 'basic.9y','basic',data['education'])

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
	X = (X / 1000).astype(float32)
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

class NeuralNet(object): 
	def __init__(self): 
		# Generate random numbers 
		random.seed(1) 

		# Assign random weights to a 61 x 1 matrix, 
		self.synaptic_weights = 2 * random.random((61, 1)) - 1

	# The Sigmoid function 
	def __sigmoid(self, x): 
		return 1 / (1 + exp(-x)) 

	# The derivative of the Sigmoid function. 
	# This is the gradient of the Sigmoid curve. 
	def __sigmoid_derivative(self, x): 
		return x * (1 - x) 

	# Train the neural network and adjust the weights each time. 
	def train(self, inputs, outputs, training_iterations): 
		for iteration in xrange(training_iterations): 
			# Pass the training set through the network. 
			output = self.learn(inputs) 

			# Calculate the error 
			error = outputs - output 

			# Adjust the weights by a factor 
			factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
			self.synaptic_weights += factor
			# print('Độ chính xác: ',self.acc(self.learn(inputs),outputs))

		# The neural network thinks. 

	def learn(self, inputs): 
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

	def acc(self, y, yPred):
		acc = sum(y == yPred) / len(y) * 100
		return acc


if __name__ == "__main__": 

	print('=======================================================')
	print('------------Percentage------------')
	data = readDataset()
	subscriptionPercent(data)
	print('=======================================================')
	print('------------Oversampling - SMOTE------------')
	# ANN
	data_final = numberVariable(data)
	X_train,y_train,X_test,y_test = createSmote(data_final)

	X_train = array(X_train)
	y_train = array(y_train)

	X_test = array(X_test)
	y_test = array(y_test)

	# Initialize 
	neural_network = NeuralNet()

	# The training set.

	# Train the neural network 
	neural_network.train(X_train, y_train, 400)

	# Test the neural network with a test example.
	print(neural_network.learn(X_train))


