# ft_linear_regression
Linear Regression Machine Learning Model to Predict a Car's Price Based on Mileage

A graphical representation to show the relationship between theta0(w), theta1(b) and the cost<br><br>
![image](https://github.com/user-attachments/assets/865ae349-f3b3-4ff4-8e84-1c786d1a8e11)
The raw data and model fit is shown on the right.


## How To Run
Clone repo<br>
```git clone git@github.com:0bada1/ft_linear_regression.git```<br>
Add your csv file with the name ```data.scv``` to this data repository at ft_linear_regression/data (Note this is a single feature linear regression model)<br>
```cd ft_linear_regression/train_model/```<br>
Train model<br>
```./train_model.sh```<br>
```cd ../predict/```<br>
Run ↓ to predict car price. Enter car mileage in \[mileage]<br>
```py3 predict.py [mileage]```
```cd ../precision/```<br>
Run ↓ to check precision<br>
```py3 precision.py```

### Developed solo by: [Obada Outabachi](https://github.com/0bada1)
