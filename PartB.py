# Rey Baltar, hrb217

## PART B PROGRAM

##     SOURCES
##https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
##https://www.kaggle.com/aasimohyeah/linear-regression-gradient-descent-on-iris-dataset
##https://medium.com/analytics-vidhya/gradient-descent-from-scratch-understanding-implementing-the-algorithm-on-boston-dataset-9d916b89d697 

#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#cost function
def  cal_cost(theta,X,y):
    '''
    
    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas 
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))
    
    where:
        j is the no of features
    '''
    
    m = len(y)
    
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost
#gradient descent function

def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        
        prediction = np.dot(X,theta)
        
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history

#stochastic
def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    
    
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost
        
    return theta, cost_history
#mini batch
def minibatch_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size =20):
    '''
    X    = Matrix of X without added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m/batch_size)
    
    for it in range(iterations):
        cost =0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            
            X_i = np.c_[np.ones(len(X_i)),X_i]
           
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost
        
    return theta, cost_history
## MAIN##
plt.style.use(['ggplot'])

#test data (debug)

##X = 2 * np.random.rand(100,1)
##y = 4 +3 * X+np.random.randn(100,1)

df=pd.read_csv('iris_dataset.csv')
#taking the 50 rows of Iris-versicolor species

rows = 50
X = df.sepal_length.iloc[rows:100].values.reshape(rows,1)
y = df.sepal_width.iloc[rows:100].values.reshape(rows,1)


plt.plot(X,y,'b.')

plt.xlabel("$Sepal Length$", fontsize=12)
plt.ylabel("$Sepal Width$", rotation=0, fontsize=12)
#_ =plt.axis([4,8,2,5])
plt.show()


#Linear Regression
X_b = np.c_[np.ones((rows,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("theta best:",theta_best)
y_predict = X_b.dot(theta_best)

plt.plot(X_b,y_predict,'r-')
plt.plot(X,y,'b.')
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([4,8,2,4])
plt.show()

#Batch stuffffffffffffffffffffff
lr =0.01
n_iter = 200
np.random.seed(7)
theta = np.random.randn(2,1)
print("theta: ", theta)

X_b = np.c_[np.ones((rows,1)),X]
theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)

print("Gradient  :")
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

#Batch: check iterations with data
fig,ax = plt.subplots(figsize=(8,4))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_=ax.plot(range(n_iter),cost_history,'b.')
plt.show()

###trim graph based on last graph
##fig,ax = plt.subplots(figsize=(10,8))
##_=ax.plot(range(200),cost_history[:200],'b.')
##plt.show()


# Stochastic: stufffffffffffffffffffffffffff

theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1)),X]
theta,cost_history2 = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)

print("Stochastic Gradient  :")
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history2[-1]))

# Stochastic: iterations with data
fig,ax = plt.subplots(figsize=(8,4))

ax.set_ylabel('{J(Theta)}',rotation=0)
ax.set_xlabel('{Iterations}')
_=ax.plot(range(n_iter),cost_history2,'--r' )
plt.show()


# Mini Batch: stufffffffffffffffffffffffff


theta = np.random.randn(2,1)

theta,cost_history3 = minibatch_gradient_descent(X,y,theta,lr,n_iter)

print("Mini Batch Gradient  :")
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history3[-1]))
# Mini Batch: iterations with data
fig,ax = plt.subplots(figsize=(8,4))

ax.set_ylabel('{J(Theta)}',rotation=0)
ax.set_xlabel('{Iterations}')
_=ax.plot(range(n_iter),cost_history3,'g^' )
plt.show()

## all data
size = 25
fig,ax = plt.subplots(figsize=(8,6))

ax.set_ylabel('{J(Theta)}',rotation=0)
ax.set_xlabel('{Iterations}')
theta = np.random.randn(2,1)
ax.axis([-1,size,0,400])
_=ax.plot(range(size),cost_history[:size],'b.', range(size),cost_history2[:size],'r--',range(size),cost_history3[:size],'g^' )
plt.show()


