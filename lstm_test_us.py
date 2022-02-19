import deds
import deds.activation

class LSTM:

  def training(self, x, lr, epochs):

  for n in range(epochs);
    C = []
    cbar = []

    for t in range():
    #def forward(self, C_old, h_old, x, Wi, Wf, Wo, Wc):
      cbar = tanh(np.dot(W_c, x[t]) + np.dot(W_c, h_old) + b_c)
      # input gate
      i_aux = sigmoid(np.dot(W_i, x[t]) + np.dot(W_i, h_old) + b_i)
      i_g = np.multiply(i_aux, cbar)

      # forget gate
      f_aux  = sigmoid(np.dot(W_f, x[t]) + np.dot(W_f, h_old) + b_f)
      f_g = np.multiply(f_aux, C_old)

      # output gate
      o_g = sigmoid(np.dot(W_o, x[t]) + np.dot(W_o, h_old) + b_o)

      # update cell state
      C[t] = f_g + i_g

      # update h state
      h[t] = np.multiply(tanh(C), o_g)

       # return h, C


    #def backward(self): 
    for t in reversed(range()):
        # OUTPUT
        # dc_dw
        dc_dz_o = (h[t] - y[t]) # TODO: mudar para d_loss 
        gradients[j][0] += np.dot(dc_dz_o, A[i][0].T)/len(y[t])

        # dc_db
        gradients[j][1] += dc_dz_o/len(y[t])

        dc_dz_t1 = dc_dz_o


        # LSTM
        W_t1 = model[j+1][0] # weight of next layer

        # dc_dw
        dc_da = np.dot(dc_dz_t1.T, model[j+1][0]).T
        da_dw = np.dot(d_act_(A[i][1]), A[i][0].T) 
        gradients[j][0] += dc_da*da_dw

        # dc_db
        gradients[j][1] += dc_da * d_act_(A[i][1]) 

        # update param of hidden state
        gradients[j][3] += da_dw

        
def LoadText():
  #open text and return input and output data (series of words)
  with open("deds/datasets/eminem.txt", "r") as text_file:
    data = text_file.read()

  text = list(data)
  outputSize = len(text)
  data = list(set(text))
  uniqueWords, dataSize = len(data), len(data) 
  returnData = np.zeros((uniqueWords, dataSize))

  for i in range(0, dataSize):
    returnData[i][i] = 1

  returnData = np.append(returnData, np.atleast_2d(data), axis=0)
  output = np.zeros((uniqueWords, outputSize))

  for i in range(0, outputSize):
    index = np.where(np.asarray(data) == text[i])
    output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()  

  return returnData, uniqueWords, output, outputSize, data

#write the predicted output (series of words) to disk
def ExportText(output, data):
  finalOutput = np.zeros_like(output)
  prob = np.zeros_like(output[0])
  outputText = ""
  print(len(data))
  print(output.shape[0])
  for i in range(0, output.shape[0]):
    for j in range(0, output.shape[1]):
      prob[j] = output[i][j] / np.sum(output[i])
    np.nan_to_num(prob)
    outputText += np.random.choice(data, p=prob)    
  with open("output.txt", "w") as text_file:
    text_file.write(outputText)
  return


if __name__ == "__main__":

  ls = LSTM()
  h, c = ls.forward(self, C_old, h_old, x, Wi, Wf, Wg, Wc)


"""
if __name__ == "__main__":

  #Begin program    
  print("Beginning")
  iterations = 5000
  learningRate = 0.001
  #load input output data (words)
  returnData, numCategories, expectedOutput, outputSize, data = LoadText()
  print("Done Reading")
  #init our RNN using our hyperparams and dataset
  RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

  #training time!
  for i in range(1, iterations):
    #compute predicted next word
    RNN.forwardProp()
    #update all our weights using our error
    error = RNN.backProp()
    #once our error/loss is small enough
    print("Error on iteration ", i, ": ", error)
    if error > -100 and error < 100 or i % 100 == 0:
      #we can finally define a seed word
      seed = np.zeros_like(RNN.x)
      maxI = np.argmax(np.random.random(RNN.x.shape))
      seed[maxI] = 1
      RNN.x = seed  
      #and predict some new text!
      output = RNN.sample()
      print(output)    
      #write it all to disk
      ExportText(output, data)
      print("Done Writing")
  print("Complete")
 
"""
