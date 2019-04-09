from scipy.io import loadmat
import numpy as np
from sklearn.metrics import mean_squared_error


# This is the som class that outputs an azimuth and 
# elevation based on the received input data and the previously 
# learned and stored SOM weights

class SOM:   


    def __init__(self, som_path, n_evaluate):
    # som_path is the path to the previously stored data (som_weights,prefStimuli)
    # n_evaluate is the number of neurons that should be used for evaluation
        
        # read the data
        # Read the som weights (previously calculated by the matlab neural som algorithm)
        som_data = loadmat(som_path)
        # Parameters 
        self.som_weights = som_data['som']
        self.pref_stimuli = som_data['prefStimuli']
        self.n_width = int(som_data['neuronCountW'][0,0])
        self.n_height = int(som_data['neuronCountH'][0,0])
        self.labels = som_data['labels']
        self.som_data_dimensions = self.som_weights.shape[1]//2
        self.L = n_evaluate
        self.response_activity = np.zeros((self.n_width * self.n_height))


    def netInput(self,x,m=0):
        # This function depends on the netInput function used in training the SOM

         # # sigmoid input function
        # xiP =  (1 / (1 + np.exp(-(6/m)*(x-m))))
        # xiM = 1- (1 / (1 + np.exp(-(6/m)*(x-m))))

        # linear input function
        xiP = x
        xiM = 1-x

        return xiP,xiM


    def calculate_location(self,data_point):

        netOutput = np.zeros(( self.som_data_dimensions,1))
        
        # calculate the input to the net
        input_layer_out = np.zeros((self.som_data_dimensions, 2))


        # get the input layer outputs
        [input_layer_out[:,0],input_layer_out[:,1]] = self.netInput(data_point)

        # flatten the matrix so we have all values in a 1d array
        input_layer_out = input_layer_out.flatten()
        # calculate the activation of the neurons
        a = sum((input_layer_out * self.som_weights).T)
        # is equal to matlab dot of 2 matrices
        b = np.diag(self.som_weights.dot(self.som_weights.T))
        c =  input_layer_out.dot(input_layer_out.T)
        # m is the final activity of the neurons. Normalize it
        m = a / np.sqrt((b)) / np.sqrt(c)
        # get the best L neurons
        winning_node_inds = np.argpartition(m, -self.L)[-self.L:]

        # create the 2d activity map
        # (xs,ys) = np.unravel_index(winning_node_inds,(response_activity.shape[0],response_activity.shape[1]))
        # response_activity[i, ys, xs] = m[winning_node_inds]
        self.response_activity = m

        # calculate perceived stimuli
        f_perceived = np.sum(m[winning_node_inds] * self.pref_stimuli[winning_node_inds, :].T,1)  / sum(m[winning_node_inds])

        winning_node_activity = m

        return f_perceived,winning_node_activity,winning_node_inds


        

