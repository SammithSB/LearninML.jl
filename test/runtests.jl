using SML


using Test


# Define a test set for the model
@testset "SML.jl" begin
    # Define some test data
    X = [1 2; 3 4; 5 6]
    y = [0.5, 0.7, 0.9]

    # Fit the model using the fit function
    weights, bias = linear_regression(X, y, 0.01, 1000)

    # Use the predict function to make predictions on the test data
    y_pred = predict(X, weights, bias)

    # Use the @test macro to check that we got an output
    @test y_pred isa Vector
end



