# %%
#################
### Libraries ###
#################
from PIL import Image
import os, glob, shutil, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# %% 
##############################################################
### Split dataset: From original_set to train_set/test_set ###
##############################################################
"""
( 가지고 있는 데이터셋에 맞게 아래 1~2번 수작업 필요 )
1. 가지고 있는 rock, scissor, paper 모든 데이터를 \original_set(폴더 만든 후)으로 이동
2. 모든 파일명을 'blabla (###).jpg' 형태로 바꿔줄 것: 'Ctrl+A'(모든파일 선택)->'F2'(이름 바꾸기)
"""
original_dir = r"C:\Users\Jaewoong\Desktop\AIFFEL\Exploration\rock_scissor_paper\original_set"
base_dir = os.path.dirname(original_dir)
train_dir = base_dir+r"\train_set"
test_dir = base_dir+r"\test_set"
if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)

# train_set으로 original_set을 전체 복사
shutil.copytree(original_dir, train_dir)

n_test_files_per_label = 100
label_lst = ['rock', 'scissor', 'paper']

for label in label_lst:
	if not os.path.isdir(test_dir+f"\{label}"):
		os.makedirs(test_dir+f"\{label}")
	original_files = glob.glob(train_dir+f"\{label}\*.jpg")
	n_train_files_per_label = len(original_files)	- n_test_files_per_label

	# 전체 데이터셋(original_set)에서 test_set을 random sampling
	test_files = random.sample(original_files, n_test_files_per_label)
	if not set(test_files) == {0}:
		for file_to_move in test_files:
			shutil.move(file_to_move, test_dir+f"\{label}")
	
total_num_train_set = n_train_files_per_label*len(label_lst)
total_num_test_set = n_test_files_per_label*len(label_lst)

# %% 
#####################
### Resize images ###
#####################
def resize_images(img_path):
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")
	
# 각 dataset 이미지가 저장된 디렉토리 아래의 모든 jpg 파일 resizing
for label in label_lst:
	train_img_path = train_dir+f"\{label}"
	test_img_path = test_dir+f"\{label}"
	resize_images(train_img_path)
	print(f"The size of {label.upper()} train_set is 28x28!")

	resize_images(test_img_path)
	print(f"The size of {label.upper()} test_set is 28x28!")

# %% 
##########################
### Load train dataset ###
##########################
def load_data(img_path, number_of_data=6000):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

# load image data
(x_train, y_train)=load_data(train_dir, number_of_data=total_num_train_set)

# normalization
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

# # %% 이미지 불러와보기
# plt.imshow(x_train[8])
# print('라벨: ', y_train[8])

# %% 
############################
### Design train network ###
############################
# load image data
(x_train, y_train)=load_data(train_dir, number_of_data=total_num_train_set)

# normalization
x_train_norm = x_train/255.0

n_channel_1=32
n_channel_2=32
n_dense=64
n_train_epoch=10

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()

# %%
###################
### Train model ###
###################
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.fit(x_train_norm, y_train, epochs=n_train_epoch)

# %%
#########################
### Load test dataset ###
#########################
# test case 1 (new dataset)
test_dir_new = r"C:\Users\Jaewoong\Desktop\AIFFEL\Exploration\data\rock_scissor_paper\rock_scissor_paper_test"
(x_test_new, y_test_new)=load_data(test_dir_new, number_of_data=300)
x_test_norm_new = x_test_new/255.0

# test case 2 (Split dataset)
(x_test, y_test)=load_data(test_dir, number_of_data=total_num_test_set)
x_test_norm = x_test/255.0
print('\n')

##########################
### Test trained model ###
##########################
# test case 1 (new dataset)
print('-------------------- test case 1 (New dataset) --------------------')
test_loss_new, test_accuracy_new = model.evaluate(x_test_norm_new, y_test_new, verbose=2)
print("Hyperparams... n_ch_1: {}, n_ch_2: {}, n_dense: {}, n_train_epoch: {}".format(n_channel_1, n_channel_2, n_dense, n_train_epoch))
print("test_loss: {} ".format(test_loss_new))
print("test_accuracy: {}".format(test_accuracy_new))
print('\n')

# test case 2 (Split dataset)
print('-------------------- test case 2 (Split dataset) --------------------')
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose=2)
print("Hyperparams... n_ch_1: {}, n_ch_2: {}, n_dense: {}, n_train_epoch: {}".format(n_channel_1, n_channel_2, n_dense, n_train_epoch))
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
# %%
