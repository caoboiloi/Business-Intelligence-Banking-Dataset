import pandas as pdimport numpy as npfrom sklearn import preprocessingimport matplotlib.pyplot as pltfrom sklearn.linear_model import LogisticRegressionfrom sklearn import metricsfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import accuracy_scorefrom matplotlib.image import imreadfrom sklearn.metrics import confusion_matrixfrom sklearn.feature_selection import RFEfrom imblearn.over_sampling import SMOTEfrom sklearn.metrics import roc_auc_scorefrom sklearn.metrics import roc_curveimport seaborn as snsimport warningsimport timewarnings.filterwarnings("ignore")def readDataset():	# Đọc dataset	data = pd.read_csv("BankAdditional/bank-full.csv")	data = data.dropna()	# Row: 41188 record - col: 21 fields	# print(data.shape)	# Tiêu đề	# print(list(data.columns))	# In 5 dòng đầu	# print(data.head())	# Dữ liệu cột education	print(data['education'].unique())	# Nhóm dữ liệu basic.4y - basic.6y - basic.9y thành basic	data['education'] = np.where(data['education'] == 'basic.4y','basic',data['education'])	data['education'] = np.where(data['education'] == 'basic.6y','basic',data['education'])	data['education'] = np.where(data['education'] == 'basic.9y','basic',data['education'])	# Kiểm tra:	# print(data['education'].unique())	return data# Tỉ lệ phần trăm của khách hàng đăng ký và không đăng kýdef subscriptionPercent(data):	# Tỉ lệ phần trăm những khách hàng đăng ký	count = len(data['y'])	count_sub = len(data[data['y']==0])	count_no_sub = len(data[data['y']==1])	sub_percent = count_sub/count	no_sub_percent = count_no_sub/count	print('tỉ lệ phần trăm khách hàng không đăng ký:',round(no_sub_percent * 100,2),'%')	print('tỉ lệ phần trăm khách hàng đăng ký:',round(sub_percent * 100,2),'%')    # Trung bình theo biến phân loại y:    # print(data.groupby('y').mean())def numberVariable(data):	# Chuyển đổi biến chuỗi thành các cột chứa dữ liệu dạng binary (có hoặc không)	# vd: marital : married -> marital_married : 1 nếu khách hàng đó có kết hôn hay marital_married : 0 nếu khách hàng ko có kết hôn	vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']	for var in vars:	    cat_list = 'var' + '_' + var	    cat_list = pd.get_dummies(data[var], prefix = var)	    data1 = data.join(cat_list)	    data = data1	vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']	data_vars = data.columns.values.tolist()	to_keep = [i for i in data_vars if i not in vars]	data_final = data[to_keep]	# Test title dataset	# print(data_final.columns.values)	return data_final# xử lý dữ liệu: do dataset majority chiếm phần lớn nên ta phải cân bằng minority có lượng dữ liệu như nhau.# Sau đó train mô hình ở tập dữ liệu mới này.def createSmote(data_final):	# Oversampling: 	# Tạo dữ liệu giả cho tập minority sao cho số phần tử của nó được nhiều lên.	# Cách đơn giản nhất là lặp lại mỗi điểm trong minority nhiều lần.	# tránh trường hợp khi phân training test chỉ tách ra majority -> gây sai lệch kết quả kiểm tra độ chính xác	# ================================================================	# SMOTE:	# Với mỗi điểm trong tập minority, tìm k điểm cùng trong minority gần nó nhất rồi dùng tổng có trọng số của các điểm này để tạo ra các điểm dữ liệu mới. 		# Test new dataset	# print(data_final)	# Lấy tất cả biến bỏ biến phân loại y	X = data_final.loc[:, data_final.columns != 'y']	# Lấy biến phân loại y	y = data_final.loc[:, data_final.columns == 'y']	os = SMOTE(random_state=0)	# Phân data thành 2 phần: training data và testing data -> 30% Test - 80% train	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)	# test length Training data, Testing data	print('Số lượng tập train:',len(X_train))	print('Số lượng tập test:',len(X_test))	# title dataset	feature_rol = X_train.columns	# dữ liệu được tạo ra hơn 10000 điểm dữ liệu mới	os_data_X,os_data_y = os.fit_sample(X_train, y_train)	os_data_X = pd.DataFrame(data = os_data_X,columns = feature_rol)	os_data_y = pd.DataFrame(data = os_data_y,columns = ['y'])	# Kiểm tra dữ liệu	print('length Oversampling:',len(os_data_X))	print('Số lượng khách hàng không đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==0]))	print('Số lượng khách hàng đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==1]))	print('Tỉ lệ khách hàng không đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==0])/len(os_data_X))	print('Tỉ lệ khách hàng đăng ký trong tập Oversampling:',len(os_data_y[os_data_y['y']==1])/len(os_data_X))	# Dữ liệu sau khi Oversampling dựa trên dữ liệu tập train data, do đó không có thông tin nào dựa vào tập test data	# Kết quả dự đoán sẽ được chính xác	return os_data_X,os_data_y,X_test,y_testclass NeuralNetwork:    ''' Khởi tạo một Neural network xử lý nhận diện số viết tay'''    '''Neural Network với input output layer va 3 hidden layer lần lượt có các số node 61, 61, 61, 61, 1 '''        def __init__(self, X, Y, layers=[61,61,61,1], lr=0.00001, iterations=100, epoches=20):        # Learning rate (lr) là 0.00001 và số lần learn (iterations) là 100        self.X = X                self.epoches = epoches                # Tạo các batch         self.batches = []        # mỗi batch chứa size là 51134/100 = 511        #         batch_size = round(len(X)/iterations)        print(len(X))        # Tạo mảng chứa index từ 0 đến 51134        idxRange = np.array(range(len(X)))        for i in range(iterations):            # Chọn ngẫu nhiên các index và bỏ vào mỗi batch            self.batches.append(np.random.choice(idxRange,batch_size,replace=False))            # Xóa các index đã chọn            idxRange = np.delete(idxRange,np.where(idxRange == self.batches[-1]))        self.Y = Y        self.rawY = Y        self.lr = lr        self.layers = layers        self.iterations = iterations        self.loss = np.array([])        self.init_weights()# Khởi tạo ngẫu nhiên weight và bias ban đầu    def init_weights(self):        self.W1 = np.random.randn(61, self.layers[0])         self.W2 = np.random.randn(self.layers[0],self.layers[1])        self.W3 = np.random.randn(self.layers[1],self.layers[2])         self.W4 = np.random.randn(self.layers[2],self.layers[3])         self.b1 = np.random.randn(self.layers[0],)        self.b2 = np.random.randn(self.layers[1],)        self.b3 = np.random.randn(self.layers[2],)        self.b4 = np.random.randn(self.layers[3],)    # Hàm sigmoid    def sigmoid(self,Z):        return 1.0/(1.0+np.exp(-Z))        # Hàm cost    def cost(self,yPred,Y):        return -(Y*np.log(yPred)+(1-Y)*np.log(1-yPred))        # Forward Propagation       def forward_propagation(self,X,Y):        Z1 = np.dot(X,self.W1)+self.b1        a1 = self.sigmoid(Z1)        Z2 = np.dot(a1,self.W2)+self.b2        a2 = self.sigmoid(Z2)        Z3 = np.dot(a2,self.W3)+self.b3        a3 = self.sigmoid(Z3)        Z4 = np.dot(a3,self.W4)+self.b4        yPred = self.sigmoid(Z4)        self.Z1 = Z1        self.Z2 = Z2        self.Z3 = Z3        self.Z4 = Z4        self.a1 = a1        self.a2 = a2        self.a3 = a3        return yPred, self.cost(yPred,Y)        # Back Propagation     def back_propagation(self,x,y,yPred):                # Đạo hàm hàm sigmoid        def dSigmoid(Z):            return self.sigmoid(Z)*(1-self.sigmoid(Z))                        # wrt là "with respect to"        # Đạo hàm hàm Cost với z tại layer thứ 4        dC_wrt_z4 = yPred-y        # Đạo hàm hàm Cost với weight tại layer thứ 4        dC_wrt_w4 = self.a3.T.dot(dC_wrt_z4)        # Đạo hàm hàm Cost với bias tại layer thứ 4        dC_wrt_b4 = np.sum(dC_wrt_z4, axis=0)        # Đạo hàm hàm Cost với a tại layer thứ 3        dC_wrt_a3 = dC_wrt_z4.dot(self.W4.T)                dC_wrt_z3 = dC_wrt_a3 * dSigmoid(self.Z3)        dC_wrt_w3 = self.a2.T.dot(dC_wrt_z3)        dC_wrt_b3 = np.sum(dC_wrt_z3, axis=0)        dC_wrt_a2 = dC_wrt_z3.dot(self.W3.T)                dC_wrt_z2 = dC_wrt_a2 * dSigmoid(self.Z2)        dC_wrt_w2 = self.a1.T.dot(dC_wrt_z2)        dC_wrt_b2 = np.sum(dC_wrt_z2, axis=0)        dC_wrt_a1 = dC_wrt_z2.dot(self.W2.T)        dC_wrt_z1 = dC_wrt_a1 * dSigmoid(self.Z1)        dC_wrt_w1 = x.T.dot(dC_wrt_z1)        dC_wrt_b1 = np.sum(dC_wrt_z1, axis=0)                # Cập nhật weight và bias bằng Gradient Descend        self.W1 = self.W1 - self.lr * dC_wrt_w1        self.W2 = self.W2 - self.lr * dC_wrt_w2        self.W3 = self.W3 - self.lr * dC_wrt_w3        self.W4 = self.W4 - self.lr * dC_wrt_w4                        self.b1 = self.b1 - self.lr * dC_wrt_b1        self.b2 = self.b2 - self.lr * dC_wrt_b2        self.b3 = self.b3 - self.lr * dC_wrt_b3        self.b4 = self.b4 - self.lr * dC_wrt_b4        # Hàm để fit data    def fit(self):        # Lấy thời gian bắt đầu thuật toán        start_time = time.time()                print("Epoches: ",self.epoches," Iterations: ", self.iterations, " Lr: ",self.lr)        self.init_weights()                        for i in range(self.epoches):                        # Mỗi batch là mỗi iterations            for j in range(self.iterations):                batch = self.batches[j]                # Thực hiện forward propagation                yPred, cost = self.forward_propagation(self.X[batch], self.Y[batch])                # Lưu lại giá trị loss : hàm cost (giá trị mất mát)                self.loss = np.append(self.loss,cost)                # Thực hiện backward propagation                self.back_propagation(self.X[batch], self.Y[batch], yPred)                        print("Epoch ",i+1," finished",end='')            # Tính accuracy            print(" Accuracy:", round(self.acc(self.rawY,self.predict(self.X))[0],2),"%",end='')            print(" Finished in: ",time.time()-start_time," seconds")                        start_time = time.time()                    print("Tranning finished")        # Hàm predict data    def predict(self, x):        Z1 = x.dot(self.W1) + self.b1        A1 = self.sigmoid(Z1)        Z2 = A1.dot(self.W2) + self.b2        A2 = self.sigmoid(Z2)        Z3 = A2.dot(self.W3) + self.b3        A3 = self.sigmoid(Z3)        Z4 = A3.dot(self.W4) + self.b4        pred = self.sigmoid(Z4)        tempPred = []        for i in pred:        	tempPred.append(i[0])        self.plotPred = tempPred        return np.round(pred)    def one_predict(self, x):    	Z1 = x.dot(self.W1) + self.b1    	A1 = self.sigmoid(Z1)    	Z2 = A1.dot(self.W2) + self.b2    	A2 = self.sigmoid(Z2)    	Z3 = A2.dot(self.W3) + self.b3    	A3 = self.sigmoid(Z3)    	Z4 = A3.dot(self.W4) + self.b4    	pred = self.sigmoid(Z4)    	return pred        # Vẽ hàm loss    def plot_loss(self):        plt.plot(self.loss)        plt.xlabel("Iteration")        plt.ylabel("logloss")        plt.title("Độ mất mát của dữ liệu")        plt.savefig('plot_loss_mlp')    def plot_predict(self):    	predicted = self.plotPred    	fig, ax = plt.subplots()    	tempTrue = []    	tempFalse = []    	predT = []    	predF = []    	for i in range(0,len(self.rawY)):    		if round(predicted[i]) == 1:    			tempTrue.append(i)    			predT.append(predicted[i])    		elif round(predicted[i]) == 0:    			tempFalse.append(i)    			predF.append(predicted[i])    	ax.plot([0, 1], [0.5, 0.5], 'k--', lw = 4)    	ax.scatter(tempTrue, predT, s = 2)    	ax.scatter(tempFalse, predF, s = 2)    	    	ax.set_xlabel('Số lượng dữ liệu')    	ax.set_ylabel('Xác suất dự đoán')    	ax.set_title('Sự phân bố xác suất dự đoán của tập dữ liệu')    	plt.savefig('plot_predict_mlp')        # Tính độ chính xác    def acc(self, y, yPred):        acc = sum(y == yPred) / len(y) * 100        return accprint('=======================================================')print('------------Percentage------------')data = readDataset()subscriptionPercent(data)print('=======================================================')print('------------Oversampling - SMOTE------------')data_final = numberVariable(data)X_train,y_train,X_test,y_test = createSmote(data_final)print('=======================================================')print('------------Multilayer Perceptron------------')# MLPX_train = np.array(X_train)y_train = np.array(y_train)X_test = np.array(X_test)y_test = np.array(y_test)# Khởi tạo neural Networknet = NeuralNetwork(X_train,y_train)# Trainingnet.fit()print('=======================================================')print('------------Plot Loss------------')# Đồ thị mất mát của dữ liệu# net.plot_loss()print("SUCCESS")print('=======================================================')print('------------Plot Predict------------')net.plot_predict()print("SUCCESS")print('=======================================================')print('------------Testing Data------------')# random 10 tập ngẫu nhiên so sánh kết quả từ tập Testindx = [np.random.randint(0,len(X_test)) for i in range(10)]temp = 0for i in indx:    check = X_test[i]    print("___________________________________________")    print("Vị trị dữ liệu dự đoán: ",i)    print("Dự đoán: ",round(net.one_predict(check)[0]))    print("Kết quả đúng: ",y_test[i][0])    if round(net.one_predict(check)[0]) == y_test[i]:    	temp += 1print('Độ chính xác dự đoán trong 10 tập ngẫu nhiên: ',round(temp/10 *100,2),'%')