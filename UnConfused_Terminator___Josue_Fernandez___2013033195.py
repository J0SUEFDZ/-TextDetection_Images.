from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
# Change data_home to wherever to where you want to download your data
mnist = fetch_mldata('MNIST original')

#-------------------------Hyperparametros-------------------------#

#Por lo tanto, en la primera iteracion se tomarán, con batch = 20:
#Iteracion  ini fin
#1          0   20
#2          20  40
#3          40  60
#Se irá sumando batch a ini y fin de forma que se tomen todos los datos
layers = 1 #Cantidad de capas ocultas, puede ser 1 o 2
layersize_1 = 10 #tamaño de la primera capa oculta
layersize_2 = 128 #tamaño de la segunda capa oculta, en caso de que layers=2
modelo = {}
guardar = False #Si se desea guardar el modelo al finalizar
dropout = 0
data = mnist.data

#---------------------Generacion de Pesos---------------------
def generate_Weights(modelo, cargar=False):
    if cargar:
        with open('save.p', 'rb') as handle:
            modelo = pickle.load(handle)
    else:
        #Las son de 28*28)=784, aplicando la Xavier Initialization
        modelo['W1'] = np.random.randn(784, layersize_1) / np.sqrt(784) #"Xavier" initialization
        if(layers == 1):
            modelo['W2'] = np.random.randn(layersize_1,10) / np.sqrt(layersize_1)
        if(layers == 2):
            modelo['W2'] = np.random.randn(layersize_1,layersize_2) / np.sqrt(layersize_1)
            modelo['W3'] = np.random.randn(layersize_2,10) / np.sqrt(layersize_2)
generate_Weights(modelo)
#-----------------------Generar Labels-----------------------
def generate_Labels(y):
    size = y.shape[0]
    zero = np.zeros((size,10))
    for i in range(size):
        zero[i][int(y[i])]=1
    return zero
labels = generate_Labels(mnist.target)
#-----------------------Guardar Modelo-----------------------
def save():
    with open('save.p', 'wb') as handle:
        pickle.dump(modelo, handle, protocol=pickle.HIGHEST_PROTOCOL)

def Predict_Image(X,y):
    a=1
    
def sigmoid(s):
    # activation function 
    return 1/(1+np.exp(-s))
def devsigmoid(s):
    #derivative of sigmoid
    return s * (1 - s)
#Codigo del profesor
#---------------------------Softmax---------------------------#
def predict(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)
#----------Perdida----------#
def loss(X,y):
    return -np.multiply(y,np.log(predict(X)))

def train(batch):
    #batch = 100 #Cantidad de datos por epoch
    ini = 0 #Limite inicial de elementos tomados
    fin = batch #Se toman datos desde ini hasta fin
    cont = 0
    #Forward
    while(fin<=data.shape[0]):
        X = data[ini:fin]
        y = labels[ini:fin]
        z = X.dot(modelo['W1'])
        a1 = sigmoid(z)
        z2 = a1.dot(modelo['W2'])
        ini+=batch
        fin+=batch
        if(layers==1):
            softmax = predict(z2)
            #------------Calcular perdida------------#
            print("Epoch:",cont," Perdida: ",np.sum(loss(z2,y))/batch)
            
            #------------BackPropagation------------#
            error = softmax - y
            delta3 = error * devsigmoid(softmax)

            z2_error = delta3.dot(modelo['W2'].T)
            z2_delta = z2_error * devsigmoid(z2)

            modelo['W2'] -= z2.T.dot(delta3)
            modelo['W1'] -= (X.T).dot(z2_delta)
        cont +=1
    if(guardar):
        save()
train(1000)

