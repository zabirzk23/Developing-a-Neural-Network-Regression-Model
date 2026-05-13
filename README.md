# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1861" height="926" alt="image" src="https://github.com/user-attachments/assets/b502b532-0864-4835-a7a2-2b8495bcd893" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Mohamed Zabir Khan A

### Register Number:212224230162

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn. Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'Loss': []}
  def forward(self , x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3 (x)
    return x
       



ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
     for epoch in range(epochs):
        optimizer.zero_grad()   

        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['Loss'].append(loss.item())

        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
<img width="458" height="288" alt="image" src="https://github.com/user-attachments/assets/b6e46819-03dc-4398-bd42-466cdbb1fbde" />


### OUTPUT
<img width="1751" height="780" alt="Screenshot 2026-04-20 144434" src="https://github.com/user-attachments/assets/e28c520a-b983-472f-92fe-65e0b1fe8cad" />

### Training Loss Vs Iteration Plot
Include your plot here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
