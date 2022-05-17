# Written by Sebastian Matiz

# Manually implement a convolutional NN using Numpy (no automatic gradient). Train and 
# test on MNIST dataset. The 93% accuracy should be achieved.  

import time
import numpy as np
import math
import sys
sys.path.append('../hw1') 
import download_mnist as mnist
import matplotlib.pyplot as plt


# 2-D Convolution Classes and Function
############################################################################
# Kernel/Filter class
class Kernel: 
    def __init__(self, dim, inDim):
        self.kernel = np.random.normal(loc=0.0, scale=(.005), size=(dim, dim))

# Convolution Layer Class   
class ConvLayer:
    def __init__(self, inCDim, outCDim, kernelDim, biasDim, inDim):
        self.kernelList = []
        self.inC = []
        self.outC = []
        self.bias = []
        self.inCDim = inCDim
        self.outCDim = outCDim
        for x in range(outCDim):
            self.bias.append(np.random.normal(loc=0.0, scale=(.005), size=(biasDim, biasDim)))
        for x in range(outCDim):
            kernelPerInput = []
            for y in range(inCDim):
                currK = Kernel(kernelDim, inDim)
                kernelPerInput.append(currK)
            self.kernelList.append(kernelPerInput)

# Max Pooling Class
class MaxPoolingLayer:
    def __init__(self, kernelDim, stride):
        self.kernelDim = kernelDim
        self.inC = []
        self.outC = []
        self.stride = stride

# Leaky Relu for Convolution Layer class
class ConvLReLu:
    def __init__(self):
        self.inC = []
        self.outC = []

# Leaky Relu for Convolution function
def convLReluFunc(convLRelu):
    convLRelu.outC = []
    for x in convLRelu.inC:
        res = np.full(x.shape, 0.0, dtype=float)
        relu = lambda t: max(0.0001, t)
        vRelu = np.vectorize(relu)
        res = vRelu(x)
        convLRelu.outC.append(res)
    return res

# Maxpool helper function
def maxPool(inputMat, kernelDim, stride):
    inShape = inputMat.shape
    outputHeight = (inShape[0] - kernelDim)/stride + 1
    outputWidth = (inShape[1] - kernelDim)/stride + 1
    output = np.full((int(outputHeight), int(outputWidth)), 0, dtype=float)
    for i in range(int(outputHeight)):   
        for j in range(int(outputWidth)):
            currMat = inputMat[i*stride:i*stride+kernelDim,j*stride:j*stride+kernelDim]   
            maxVal = np.amax(currMat);
            output[i,j] = maxVal
    return output

# Maxpool function
def maxPoolForward(maxPoolingLayer):
    maxPoolingLayer.outC = []
    for x in maxPoolingLayer.inC:
        maxPoolingLayer.outC.append(maxPool(x, maxPoolingLayer.kernelDim, maxPoolingLayer.stride))

# Convolution helper function
def convolve(inputMat, kernel, stride):
    inShape = inputMat.shape
    if len(kernel.shape) == 1:
        kernel = np.reshape(kernel, (kernel.shape[0], 1))
    kShape = kernel.shape 
    outputHeight = (inShape[0] - kShape[0])/stride + 1
    outputWidth = (inShape[1] - kShape[0])/stride + 1
    output = np.full((int(outputHeight), int(outputWidth)), 0.0)
    for i in range(int(outputHeight)):   
        for j in range(int(outputWidth)):
            currMat = inputMat[i*stride:i*stride+kShape[0],j*stride:j*stride+kShape[1]]   
            conv = np.multiply(currMat, kernel)
            conv = np.sum(conv)
            output[i,j] = conv
    return output

# Full Convolution function
def fullConvolve(inputMat, kernel):
    kShape = kernel.shape
    padding = kShape[0]-1
    paddedIn = np.pad(inputMat, padding, 'constant', constant_values=(0))
    return convolve(paddedIn, kernel, 1)


# Convolution function
def convForward(convLayer, stride):
    convLayer.outC = []
    outCLength = convLayer.outCDim
    inCLength = convLayer.inCDim
    for i in range(outCLength):
        currOut = np.array([], dtype=float)
        for j in range(inCLength):
            # print ("inC[j]:\n", convLayer.inC[j])
            # print ("kernel[i,j]:\n", convLayer.kernelList[i][j].kernel)
            if len(currOut) == 0:
                currOut = convolve(convLayer.inC[j], convLayer.kernelList[i][j].kernel, stride)
            else:
                currOut = np.add(currOut, convolve(convLayer.inC[j], convLayer.kernelList[i][j].kernel, stride))
        convLayer.outC.append(currOut)

# assuming kernelDim is same for length and width
# function for finding gradient of maxpool operation
def maxPoolGrad(inputMat, outputMat_grad, kernelDim, stride):
    gradHeight = inputMat.shape[0]
    gradWidth = inputMat.shape[1]
    grad = np.full((int(gradHeight), int(gradWidth)), 0.0)
    loopIRange  = (inputMat.shape[0] - kernelDim)/stride + 1
    loopJRange = (inputMat.shape[1] - kernelDim)/stride + 1
    for i in range(int(loopIRange)):
        for j in range(int(loopJRange)):
            currMat = inputMat[i*stride:i*stride+kernelDim,j*stride:j*stride+kernelDim]   
            maxIndex = np.argmax(currMat)
            currMatGrad = np.full((1, kernelDim*kernelDim), 0.0)
            currMatGrad[0,maxIndex] = 1.0*outputMat_grad[i,j]
            grad[i*stride:i*stride+kernelDim,j*stride:j*stride+kernelDim] = np.reshape(currMatGrad, (kernelDim, kernelDim))
    return grad

# adding bias for convolution channels function
def addBias(convLayer):
    for x in range(len(convLayer.outC)):
        convLayer.outC[x] = np.add(convLayer.outC[x], convLayer.bias[x])
############################################################################

# Fully Connected Layer Class and Functions
############################################################################
# Fully Connected Layer Class
class FCLayer:
    def __init__(self, numNodes, numWeightsPerNode):
        self.nodeValues = np.full((numNodes, 1),0, dtype=float)
        # intitialize weights and bias as a Gaussian distribution with mean = 0 and standard deviation sqrt(2/numNodes)
        self.bias = np.random.normal(loc=0.0, scale=(math.sqrt(2/(numNodes))), size=(numNodes, 1))
        self.weights = np.random.normal(loc=0.0, scale=(math.sqrt(2/(numNodes))), size=(numNodes, numWeightsPerNode))

# Relu Function
def LRelu(i):
    res = np.full(i.nodeValues.shape, 0.0, dtype=float)
    relu = lambda t: max(0.0001, t)
    vRelu = np.vectorize(relu)
    res = vRelu(i.nodeValues)
    return res

# Softmax Function
def softmax(vector):
    x = np.exp(vector - np.max(vector))
    return x/x.sum()

# Forward propagation for fully connected layers
def FCForward(data):
    fcOne.nodeValues = np.array(data)
    fcOne.nodeValues = np.reshape(fcOne.nodeValues, (len(data), 1))
    global LR1
    LR1 = LRelu(fcOne)
    fcTwo.nodeValues = np.matmul(fcOne.weights.T, LR1) + fcTwo.bias
    global LR2
    R2 = LRelu(fcTwo)
    outputLayer.nodeValues = np.matmul(fcTwo.weights.T, LR2) + outputLayer.bias
    outputLayer.nodeValues = softmax(outputLayer.nodeValues)

# Gradients of Softmax Function 
def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# Gradient of Leaky Relu Function
def backwardLRelu(i):
    res = np.full(i.shape, 0.0, dtype=float)
    for x in range(len(i)):
        for y in range(len(i[0])):
            res[x,y] = 0 if i[x,y] <= 0.0001 else 1
    return res
############################################################################

# Fully Connected Layers Backpropagation
# Back propagate loss and update weights + bias for each layer
def backwardFC(y_pred, y_actual, lr):
    dL_L = 1
    dL_sm = dL_L*2*(y_pred - y_actual)
    dL_out = np.matmul(dL_sm.T, softmax_grad(outputLayer.nodeValues)) # 1 x 10
    dL_LR2 = np.matmul(dL_out, fcTwo.weights.T)  # 1 x 50
    dL_W_fc2 = np.matmul(dL_out.T, LR2.T)  # 50 x 10
    dL_B_out = dL_out
    outputLayer.bias = np.subtract(outputLayer.bias, lr*dL_B_out.T)
    fcTwo.weights = np.subtract(fcTwo.weights, lr*dL_W_fc2.T)
    dL_fc2 = np.multiply(dL_LR2.T,backwardLRelu(LR2)) # 50 x 1
    dL_W_fc1 = np.matmul(dL_fc2, LR1.T) 
    dL_B_fc2 = dL_fc2
    fcTwo.bias = np.subtract(fcTwo.bias, lr*dL_B_fc2)
    fcOne.weights = np.subtract(fcOne.weights, lr*dL_W_fc1.T)
    dL_LR1 = np.matmul(dL_fc2.T, fcOne.weights.T)
    dL_fcOne = np.multiply(dL_LR1.T, backwardLRelu(LR1))
    return dL_fcOne

# Convolutional Layers Backpropagation
def backwardConv(grad, lr):
    # for the CNN, grad should be a 120 x 1 vector

    # solving gradient for input in conv3
    dx3 = []
    for i in range(len(convThree.inC)):
        currDx3 = np.full(convThree.inC[0].shape, 0.0)
        for j in range(len(grad)):
            currKernel = convThree.kernelList[j][i].kernel
            currKernel = np.fliplr(currKernel)
            currKernel = np.flipud(currKernel) 
            dIn = convolve(currKernel, grad[j], 1)
            currDx3 = np.add(currDx3, dIn)
        dx3.append(currDx3)

    # solving gradient for kernels in conv3
    for i in range(len(grad)):
        for j in range(len(convThree.inC)):
            dKernel = convolve(convThree.inC[j], grad[i], 1)
            # if i == 0:
            #     print ("X: \n", convThree.inC[j])
            #     print ("grad: \n", grad[i])
            #     print ("kernel: \n", convThree.kernelList[i][j].kernel)
            convThree.kernelList[i][j].kernel = np.subtract(convThree.kernelList[i][j].kernel, lr*dKernel)
            # if i == 0:
            #     print ("new kernel: \n", convThree.kernelList[i][j].kernel )

    # solving gradient for bias in conv3
    for i in range(len(convThree.bias)):
        convThree.bias[i] = np.subtract(convThree.bias[i], lr*grad[i])

    # solving local gradient for maxpool layer
    dMaxPool2 = []
    for i, j in zip(maxPoolTwo.inC, dx3):
        dMaxPool2.append(maxPoolGrad(i,j,2,2))

    # local gradient for ReluTwo
    dLReluTwo = []
    for x in convLReluTwo.outC:
        dLReluTwo.append(backwardLRelu(x))

    # global gradient of Loss WRT Relu
    dL_dLReluTwo = []
    for r, m in zip(dLReluTwo ,dMaxPool2):
        dL_dLReluTwo.append(np.multiply(r, m))
    
    # solving gradient for input in conv2
    dx2 = []
    for i in range(len(convTwo.inC)):
        currDx2 = np.full(convTwo.inC[0].shape, 0.0)
        for j in range(len(dL_dLReluTwo)):
            currKernel = convTwo.kernelList[j][i].kernel
            currKernel = np.fliplr(currKernel)
            currKernel = np.flipud(currKernel) 
            dIn = fullConvolve(currKernel, dL_dLReluTwo[j])
            currDx2 = np.add(currDx2, dIn)
        dx2.append(currDx2)

    # solving gradient for kernels in conv2
    for i in range(len(dL_dLReluTwo)):
        for j in range(len(convTwo.inC)):
            dKernel = convolve(convTwo.inC[j], dL_dLReluTwo[i], 1)
            convTwo.kernelList[i][j].kernel = np.subtract(convTwo.kernelList[i][j].kernel, lr*dKernel)

    # solving gradient for bias in conv2
    for i in range(len(convTwo.bias)):
        convTwo.bias[i] = np.subtract(convTwo.bias[i], lr*dMaxPool2[i])

    # solving local gradient for maxpool layer
    dMaxPool1 = []
    for i, j in zip(maxPoolOne.inC, dx2):
        dMaxPool1.append(maxPoolGrad(i,j,2,2))

     # local gradient for ReluOne
    dLReluOne = []
    for x in convLReluOne.outC:
        dLReluOne.append(backwardLRelu(x))

    # global gradient of Loss WRT Relu
    dL_dLReluOne = []
    for r, m in zip(dLReluOne ,dMaxPool1):
        dL_dLReluOne.append(np.multiply(r, m))

    # solving gradient for kernels in conv1
    for i in range(len(dL_dLReluOne)):
        for j in range(len(convOne.inC)):
            dKernel = convolve(convOne.inC[j], dL_dLReluOne[i], 1)
            convOne.kernelList[i][j].kernel = np.subtract(convOne.kernelList[i][j].kernel, lr*dKernel)
    
    # solving gradient for bias in conv1
    for i in range(len(convOne.bias)):
        convOne.bias[i] = np.subtract(convOne.bias[i], lr*dL_dLReluOne[i])

# load data
x_train, y_train, x_test, y_test = mnist.load()

data = x_train[0]
data = np.reshape(data, (28,28))
y_actual = y_train[0]
actualVal = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
actualVal[y_actual,0] = 1


# build CNN
convOne = ConvLayer(1, 6, 5, 24, 28)  # input is 28x28 output is 24 x 24
convLReluOne = ConvLReLu()
maxPoolOne = MaxPoolingLayer(2, 2)
convTwo = ConvLayer(6, 16, 5, 8, 12) # input 12 x 12 is output is 8x8
convLReluTwo = ConvLReLu()
maxPoolTwo = MaxPoolingLayer(2, 2)  
convThree = ConvLayer(16, 120, 4, 1, 4)  # input is 4x4 output is 1x1

fcOne = FCLayer(120, 84)
LR1 = np.full((120,1),0, dtype=float)
fcTwo = FCLayer(84, 10)
LR2 = np.full((84, 1),0, dtype=float)
outputLayer = FCLayer(10, 1)


# train
# print ("Input:\n", data)
def forward(input):
    convOne.inC = [input]
    convForward(convOne, 1)
    addBias(convOne)
    # print ("convOne, outC:\n")
    # for x in convOne.outC:
    #     print (x)
    #     print ()
    # print (convOne.outC[0])
    # print ()
    convLReluOne.inC = convOne.outC    
    convLReluFunc(convLReluOne)

    maxPoolOne.inC = convLReluOne.outC
    maxPoolForward(maxPoolOne)

    convTwo.inC = maxPoolOne.outC
    convForward(convTwo, 1)
    addBias(convTwo)
 
    convLReluTwo.inC = convTwo.outC    
    convLReluFunc(convLReluTwo)
    maxPoolTwo.inC = convLReluTwo.outC
    maxPoolForward(maxPoolTwo)

    convThree.inC = maxPoolTwo.outC
    convForward(convThree, 1)
    addBias(convThree)

    FCForward(convThree.outC)

# train
start_time = time.time()
np.random.seed(1)
loss = []
prevLoss = sys.float_info.max
epoch = 0
while True:
    # generating starting point for current training batch

    # varying learning rate, comment if you want static learning rate
    # if epoch != 0 and epoch % 1000 == 0:
    #     lr = lr*.5
    #     print (lr)
        
    rand = np.random.randint(0, len(y_train)-129)
    actualVal = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    currLoss = 0
    acc = 0.0
    for i, j in zip(x_train[rand:rand+128], y_train[rand:rand+128]):
        actualVal = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        actualVal[j,0] = 1
        i = np.reshape(i, (28,28))
        forward(i)
        if np.argmax(outputLayer.nodeValues) == j:
            acc += 1.0
        currLoss += np.sum(np.square(np.subtract(outputLayer.nodeValues,actualVal)))
        # print ("pred: ", np.argmax(outputLayer.nodeValues), "actual: ", np.argmax(actualVal))
        # print ()
        FCgradient = backwardFC(outputLayer.nodeValues, actualVal, .0002)
        backwardConv(FCgradient, .0002)
    loss.append(currLoss)

    if acc/128.0 > 0.97: break
    epoch += 1
    print (outputLayer.nodeValues)
    print (actualVal)
    print (epoch, " loss: ", currLoss)
    print ("accuracy: ", acc/128.0)
    print ()
print ("---training time: %s seconds ---" % (time.time() - start_time))

# test
correct = 0.0
for i,j in zip(x_test, y_test):
    i = np.reshape(i, (28,28))
    forward(i)
    if np.argmax(outputLayer.nodeValues) == j:
        correct += 1.0
correct = correct/len(y_test)
print ("Test Accuracy: ", correct)

plt.plot(loss)
plt.title("Loss Over Training")
plt.savefig("loss_plot.png")