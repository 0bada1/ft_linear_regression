# ft_linear_regression
Linear Regression Machine Learning Model to Predict a Car's Price Based on Mileage

A graphical representation to show the relationship between theta0(w), theta1(b) and the cost<br><br>
![image](https://github.com/user-attachments/assets/865ae349-f3b3-4ff4-8e84-1c786d1a8e11)
The raw data and model fit is shown on the right<br>
The contour plot shows what theta0 (w) values lead what what cost, while ignoring b for simplicity<br>
The 3D plots show how theta0 (w), theta1 (b), and cost are related. The 3D line shows the path that gradient descent took to converge to the fitting values


## How To Run
### Clone repo<br>
```git clone git@github.com:0bada1/ft_linear_regression.git```<br>
Add your csv file with the name ```data.scv``` to this data repository at ```ft_linear_regression/data``` (Note this is a single feature linear regression model)<br>
### Train model<br>
```cd ft_linear_regression/srcs/train_model/```<br>
```./train_model.sh```<br>
### Run ↓ to predict car price. Enter car mileage in \[mileage]<br>
```cd ../predict/```<br>
```py3 predict.py [mileage]```
### Run ↓ to check precision<br>
```cd ../precision/```<br>
```py3 calculate_precision.py```

### Developed solo by: [Obada Outabachi](https://github.com/0bada1)
