from keras.layers import Dense,Activation, Dropout,Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import  Sequential
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt


class ModelBuilder:
    
    def build_model(self):
        model=Sequential()
        #CONVOLUTION LAYER WITH 32 FILTERS
        model.add(Conv2D(32,(3,3),input_shape=self.INPUT_SHAPE,activation='relu',padding='same'))
        model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
        #DROPOUT LAYER TO DISCARD 25% OF WEIGHTS
        model.add(Dropout(0.25))
        #MAXPOOLING LAYER TO REDUCE THE SIZE UPTO 75%
        model.add(MaxPooling2D(pool_size=(2,2)))
        #CONVOLUTION LAYER WITH 64 FILTERS
        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        #DROPOUT LAYER TO DISCARD 25% OF WEIGHTS
        model.add(Dropout(0.25))
        #CONVOLUTION LAYER WITH 128 FILTERS
        model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
        #DROPOUT LAYER TO DISCARD 25% OF WEIGHTS
        model.add(Dropout(0.25))
        #FLATTENING TO FEED AS INPUT TO DENSE LAYER (ANN)
        model.add(Flatten())
        #DENSE LAYER WITH 256 NEURONS
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        #DENSE LAYER WITH 128 NEURONS
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(0.25))
        #DENSE LAYER WITH 10 (NO OF OUTPUT CATEGORIES) NEURONS
        model.add(Dense(self.CLASSES))
        #ACTIVATION LAYER (SOFTMAX) TO CALCULATE RESULT AS PROBABILITY OF EACH CATEGORY
        model.add(Activation('softmax'))
        return model

    
    def compile_train_model(self,model):
        model.compile(loss=self.LOSS_FUNCTION,optimizer=self.OPTIMIZER,metrics=['accuracy'])
        print(model.summary())
        history=model.fit(self.X_train,self.y_train,validation_data=(self.X_test,self.y_test),epochs=self.EPOCHS,batch_size=self.BATCH_SIZE)
        return model,history
    
    def normalize(self,x):           
        #NORMALIZING THE IMAGES TO REANGE THE PIXEL VALUES FROM 0-1
        x=x.astype('float32')
        return x/255
    
    def __init__(self):
        #LOAD THE MNIST DATASET
        (self.X_train,self.y_train),(self.X_test,self.y_test)=mnist.load_data()
        
        #STORING THE INFORMATION ABOUT THE STUCTURE OF DATASET
        self.HEIGHT=self.X_train.shape[1]
        self.WIDTH=self.X_train.shape[2]
        self.INPUT_SHAPE=(self.HEIGHT,self.WIDTH,1)
        self.CLASSES=10
        
        #PREPROCESSING THE IMAGE DATA
        self.X_train=self.normalize(self.X_train)
        self.X_test=self.normalize(self.X_test)
        self.X_train=self.X_train.reshape(self.X_train.shape[0],self.HEIGHT,self.WIDTH,1)
        self.X_test=self.X_test.reshape(self.X_test.shape[0],self.HEIGHT,self.WIDTH,1)
        
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)

        #HYPERPARAMETERS FOR THE MODEL
        self.EPOCHS=1
        self.BATCH_SIZE=128
        self.OPTIMIZER='adam'
        self.LOSS_FUNCTION='categorical_crossentropy'
        
       
    def run(self):
        #BUILDING THE MODEL
        model=self.build_model()
        #COMPILING AND TRAINING THE MODEL
        model,history= self.compile_train_model(model)
        #self.plot_loss_accuracy(history)
        #SAVING THE MODEL TO LOCAL
        model.save("model")    
        
    
    def plot_loss_accuracy(self,history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
builder=ModelBuilder()
builder.run()
        

    