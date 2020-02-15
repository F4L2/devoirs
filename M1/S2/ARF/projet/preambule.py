from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from arftools import *
from utility import *


'''USPS'''
# #(7291,256)(7291,)
train = load_usps("USPS/USPS_train.txt")
# #(2007,256)(2007,)
test = load_usps("USPS/USPS_test.txt")

# c1 = 1
# train = class_versus_all(train[0], train[1], c1)
# test = class_versus_all(test[0], test[1], c1)

'''Régressions'''

#fit régression linéaire
reg_lin = LinearRegression()
reg_lin.fit(train[0], train[1])

#predict
pred_train = reg_lin.predict(train[0])
pred_test = reg_lin.predict(test[0])


cout_train = mean_squared_error(train[1], pred_train)
cout_test = mean_squared_error(test[1], pred_test)
print("Cout \t\t train : {} | test : {}".format(cout_train, cout_test) )
print("précision \t train : {} | test : {}".format(r2_score(train[1], pred_train), r2_score(test[1], pred_test) ) )


alpha = np.linspace(0.0001, 100, 5000)
ridge_score_train = []
ridge_score_test = []
ridge_cout_train = []
ridge_cout_test = []

lasso_score_train = []
lasso_score_test = []
lasso_cout_train = []
lasso_cout_test = []

for a in alpha:
    print(a)
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(train[0], train[1])
    ridge_score_train.append(ridge.score(train[0], train[1]))
    ridge_score_test.append(ridge.score(test[0], test[1]))
    ridge_pred_train = ridge.predict(train[0]) 
    ridge_pred_test = ridge.predict(test[0])
    ridge_w = np.sqrt(sum(np.array(ridge.coef_)*np.array(ridge.coef_)))
    ridge_cout_train.append(mean_squared_error(train[1], pred_train) + a*(ridge_w**2))
    ridge_cout_test.append(mean_squared_error(test[1], pred_test) + a*(ridge_w**2))


    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(train[0], train[1])
    lasso_score_train.append(lasso.score(train[0], train[1]))
    lasso_score_test.append(lasso.score(test[0], test[1]))
    lasso_pred_train = lasso.predict(train[0]) 
    lasso_pred_test = lasso.predict(test[0])
    lasso_w = sum([abs(x) for x in lasso.coef_])
    lasso_cout_train.append(mean_squared_error(train[1], pred_train) + a*lasso_w)  
    lasso_cout_test.append(mean_squared_error(test[1], pred_test) + a*lasso_w)     
    
plt.figure()
plt.plot(alpha, ridge_score_train, marker="s", color='red', label='train')
plt.plot(alpha, ridge_score_test, marker="v", color='blue', label='test')
plt.title("Précision en fonction de alpha")
plt.legend()
plt.savefig("precision_ridge")
plt.show()

plt.plot(alpha, ridge_cout_train, marker="s", color='red', label="train")
plt.plot(alpha, ridge_cout_test, marker="v", color='blue', label="test")
plt.title("Cout en fonction d'alpha")
plt.legend()
plt.savefig("cout_ridge")
plt.show()


plt.figure()
plt.plot(alpha, lasso_score_train, marker="s", color='red', label='train')
plt.plot(alpha, lasso_score_test, marker="v", color='blue', label='test')
plt.title("Précision en fonction de alpha")
plt.legend()
plt.savefig("precision_lasso")
plt.show()

plt.plot(alpha, lasso_cout_train, marker="s", color='red', label="train")
plt.plot(alpha, lasso_cout_test, marker="v", color='blue', label="test")
plt.title("Cout en fonction d'alpha")
plt.legend()
plt.savefig("cout_lasso")
plt.show()


