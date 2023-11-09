# Heart-attack-prediction-using-MLP
## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python
## Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<br>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<br>

## Program:
```
Name: Logeshwari.P
Register number: 212221230055
```
```

import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv("/content/heart.csv")
X=data.iloc[:, :-1].values #features 
Y=data.iloc[:, -1].values  #labels 

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss=mlp.fit(X_train,y_train).loss_curve_

y_pred=mlp.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Losss")
plt.show()
```
## Output:
### X Values
![280495871-f6ccc838-e5a3-4e63-aac0-5f767145dd75](https://github.com/logeshwari2004/Experiment-6---Heart-attack-prediction-using-MLP/assets/94211349/30a0b6bd-46c4-4a18-a259-653974246f56)

### Y Values
![280495874-705aed95-1e87-44fe-97fc-5eae806099c8](https://github.com/logeshwari2004/Experiment-6---Heart-attack-prediction-using-MLP/assets/94211349/adb96b1e-64dd-4598-bf62-031463b10566)

### Accuracy
![280495877-82e6fe11-70dc-462e-a779-1aeddbf080b0](https://github.com/logeshwari2004/Experiment-6---Heart-attack-prediction-using-MLP/assets/94211349/6017ffc3-445f-44ef-acee-a12c0fd1f77e)

### Loss Convergence Graph
![280495884-c1ee3539-0ccc-46f6-8949-8246dda07612](https://github.com/logeshwari2004/Experiment-6---Heart-attack-prediction-using-MLP/assets/94211349/bc5f7c67-6659-4bce-913d-d391d880b1a8)


## Result:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     

