import numpy as np
import pickle

def load_model(file_path):
    """
    Load the trained model from a pickle file.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(model, input_data):
    """
    Make a prediction using the loaded model.
    """
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Load the model from the specified file path
    model = load_model('LR_Algorithm.pkl')
    
    # Example input data (adjust according to your model's input requirements)
    data_2 = np.array([[True, False, False, False, False, 1, 0, 0, 0, 0, 4, 98, 93, 67, 90, 74, 12, 0.99, 8]])
    
    # Make a prediction using the loaded model
    prediction = make_prediction(model, data_2)
    
    # Print the output prediction
    print("Predicted Causes of Respiratory Imbalance:", prediction)
