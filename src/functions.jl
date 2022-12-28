using LinearAlgebra

# function to fit a linear regression model
function linear_regression(X, y, learning_rate=0.01, epochs=1000, reg_lambda=0.0)
    n_samples, n_features = size(X)
    w = zeros(n_features)
    b = 0.0
    for i in 1:epochs
        # compute predictions
        y_pred = X * w .+ b

        # compute mean squared error loss
        loss = sum((y_pred .- y) .^ 2) / (2 * n_samples)
        # add regularization penalty to loss
        loss += reg_lambda * sum(w .^ 2) / (2 * n_samples)

        # compute gradients of loss with respect to weights and bias
        dw = (1 / n_samples) * X' * (y_pred .- y)
        db = (1 / n_samples) * sum(y_pred .- y)

        # add regularization term to gradients
        dw += reg_lambda * w / n_samples

        # update weights and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db
    end
    return w, b
end

# function to predict
function predict(X, w, b)
    return X * w .+ b
end



