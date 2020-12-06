from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import datetime
import os

def loadDataset( fileName, delimiter = ',' ):
    #Load dataset
    dataset = loadtxt( fileName, delimiter = delimiter )
    x = dataset[:,0:8]
    y = dataset[:,8]

    return x, y

def defineModel( inputData, outputData ):
    #Define keras model
    model = Sequential()
    model.add( Dense( 12, input_dim=8, activation='relu' ) )
    model.add( Dense( 8, activation='relu' ) )
    model.add( Dense( 1, activation='sigmoid' ) )

    #Compile the keras model
    model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

    #Fit the keras model on the dataset
    model.fit( inputData, outputData, epochs=150, batch_size=10 )

    return model

def predictOutput( inputDataSet ):
    # make class predictions with the model
    predictions = model.predict_classes( inputDataSet )
    
    # summarize the first 5 cases
    for i in range(5):
        print('%s => %d' % ( inputDataSet[i].tolist(), predictions[i]) )

def saveToFile( model ):
    fileName = datetime.datetime.now().strftime( "%Y%m%d_%H%M%S" )

    # serialize model to JSON
    model_json = model.to_json()
    with open( "trainedModels" + fileName + ".json", "w" ) as json_file:
        json_file.write( model_json )

    # serialize weights to HDF5
    model.save_weights( "trainedModels" + fileName + ".h5" )
    print( "Saved model to disk" )

def readFromFile( fileName ):
    json_file = open( 'model.json', 'r' )
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json( loaded_model_json )
    loaded_model.load_weights( 'model.h5' )
    print( 'Model loaded from file' )

    # loaded_model.compile( loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'] )

if __name__ == "__main__":
    
    print( "Run main method" )

    inputData, outputData = loadDataset( 'trainingData/trainingData.csv' )
    model = defineModel( inputData, outputData )
    saveToFile( model )
    

    #Evaluate the keras model
    # _, accuracy = model.evaluate( x, y )
    # print( 'Accurary %.2f', ( accuracy*100 ) )

