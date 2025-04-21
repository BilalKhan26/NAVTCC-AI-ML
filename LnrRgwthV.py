def univariate_linear_regression(X, y, alpha=0.01, iterations=100):
    m = len(X)
    theta_0 = 0
    theta_1 = 0

    for it in range(iterations):
        total_error_0 = 0
        total_error_1 = 0

        for i in range(m):
            prediction = theta_0 + theta_1 * X[i]
            error = prediction - y[i]
            total_error_0 += error
            total_error_1 += error * X[i]

        # Update parameters
        theta_0 -= alpha * (total_error_0 / m)
        theta_1 -= alpha * (total_error_1 / m)

    return theta_0, theta_1


X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

theta_0, theta_1 = univariate_linear_regression(X, y, alpha=0.01, iterations=100)
print(f"Learned parameters: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")

#----------------------------------------------

def univariate_linear_regression_with_tracking(X, y, alpha=0.01, iterations=100):
    m = len(X)
    theta_0 = 0
    theta_1 = 0

    for it in range(iterations):
        total_error_0 = 0
        total_error_1 = 0
        cost = 0

        for i in range(m):
            prediction = theta_0 + theta_1 * X[i]
            error = prediction - y[i]
            cost += error ** 2
            total_error_0 += error
            total_error_1 += error * X[i]

        # Calculate and print cost
        cost = cost / (2 * m)
        print(f"Iteration {it+1:3}: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}, Cost = {cost:.4f}")

        # Update parameters
        theta_0 -= alpha * (total_error_0 / m)
        theta_1 -= alpha * (total_error_1 / m)

    return theta_0, theta_1

X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

theta_0, theta_1 = univariate_linear_regression_with_tracking(X, y, alpha=0.01, iterations=10)
print(f"Learned parameters: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")
#----------------------------------------------

def mean_squared_error(X, y, theta_0, theta_1):
    m = len(X)
    total_error = 0

    for i in range(m):
        prediction = theta_0 + theta_1 * X[i]
        error = prediction - y[i]
        total_error += error ** 2

    mse = total_error / m
    return mse

X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Assume we trained the model already
theta_0, theta_1 = univariate_linear_regression_with_tracking(X, y, alpha=0.01, iterations=100)

mse = mean_squared_error(X, y, theta_0, theta_1)
print(f"\nFinal Mean Squared Error: {mse:.4f}")

#-----------------------------------------------
def predict(x, theta_0, theta_1):
    return theta_0 + theta_1 * x


X_new = 5
prediction = predict(X_new, theta_0, theta_1)
print(f"Prediction for X = {X_new}: {prediction:.4f}")
