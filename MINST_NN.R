#set.seed(123)

data <- read.csv("mnist_train.csv")
data_test <- read.csv("mnist_test.csv")

m <- nrow(data)
n <- ncol(data)
Y <- (data[, 1])
X <- t(as.matrix(data[, 2:n]))
X <- X / 255  # Simple normalization between 0 and 1

m_test <- nrow(data_test)
n_test <- ncol(data_test)
Y_test <- (data_test[, 1])
X_test <- t(as.matrix(data_test[, 2:n_test]))
X_test <- X_test / 255


#init_params <- function() {
 #   W1 <- matrix(runif(10 * 784, min = -0.5, max = 0.5), nrow = 10, ncol = 784)
 #   b1 <- matrix(runif(10, min = -0.5, max = 0.5), nrow = 10, ncol = 1)
 #   W2 <- matrix(runif(10 * 10, min = -0.5, max = 0.5), nrow = 10, ncol = 10)
 #   b2 <- matrix(runif(10, min = -0.5, max = 0.5), nrow = 10, ncol = 1)

 #   return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
#}

init_params <- function() {
  set.seed(42)  # for reproducibility
  n_hidden <- 10  # adjust as needed
  n_input <- 784
  n_output <- 10  # for MNIST
  
  W1 <- matrix(rnorm(n_hidden * n_input, mean = 0, sd = sqrt(2/n_input)), 
               nrow = n_hidden, ncol = n_input)
  b1 <- matrix(rnorm(n_hidden, mean = 0, sd = sqrt(2/n_input)), nrow = n_hidden, ncol = 1)
  W2 <- matrix(rnorm(n_output * n_hidden, mean = 0, sd = sqrt(2/n_hidden)), 
               nrow = n_output, ncol = n_hidden)
  b2 <- matrix(rnorm(n_output, mean = 0, sd = sqrt(2/n_hidden)), nrow = n_output, ncol = 1)
  
  return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}

ReLU <- function(Z) {
    return(pmax(Z, 0))
}


softmax <- function(Z) {
    # Subtract max for numerical stability, apply column-wise
    exp_Z <- exp(sweep(Z, 2, apply(Z, 2, max), "-"))
    # Normalize column-wise
    return(as.matrix(sweep(exp_Z, 2, colSums(exp_Z), "/")))
}

one_hot <- function(Y) {
    n_labels <- max(Y) + 1
    one_hot_Y <- matrix(0, nrow = n_labels, ncol = length(Y))
    # Set the appropriate elements to 1
    one_hot_Y[cbind(Y + 1, seq_along(Y))] <- 1

    return(as.matrix(one_hot_Y))
}

forward_propagation <- function(W1, b1, W2, b2, X) {
    Z1 <- sweep(W1 %*% X, 1, b1, "+")
    A1 <- ReLU(Z1)
    Z2 <- sweep(W2 %*% A1, 1, b2, "+")
    A2 <- softmax(Z2)
    return(list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2))
}

ReLU_derivative <- function(Z) {
    return(ifelse(Z > 0, 1, 0))
}

backward_propagation <- function(Z1, A1, Z2, A2, W1, W2, X, Y, m) {
    one_hot_Y <- one_hot(Y)
    dZ2 <- A2 - one_hot_Y
    dW2 <- (1 / m) * (dZ2 %*% t(A1))
    db2 <- as.matrix((1 / m) * rowSums(dZ2))
    
    ReLU_derivative_Z1 <- ReLU_derivative(Z1)
    dZ1 <- (t(W2) %*% dZ2) * ReLU_derivative_Z1
    dW1 <- (1 / m) * (dZ1 %*% t(X))
    db1 <- as.matrix((1 / m) * rowSums(dZ1))

    return(list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2))
}

update_paramaters <- function(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) {
    W1 <- W1 - alpha * dW1
    b1 <- b1 - alpha * db1
    W2 <- W2 - alpha * dW2
    b2 <- b2 - alpha * db2

    return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
get_predictions <- function(A2) {
    # Get the index (0-9) of the maximum probability for each sample
    predictions <- (max.col(t(A2))) - 1  # Subtract 1 to get 0-9 range
    return(predictions)
}


get_accuracy <- function(predictions, Y) {
    return(mean(predictions == Y))
}

gradient_descent <- function(X, Y, m, alpha, iterations) {
    params <- init_params()
    W1 <- params$W1
    b1 <- params$b1
    W2 <- params$W2
    b2 <- params$b2
    for (i in 1:iterations) {
        forward_result <- forward_propagation(W1, b1, W2, b2, X)

        #cost <- compute_cost(forward_cache$A2, Y)
        backward_result <- backward_propagation(forward_result$Z1, forward_result$A1, forward_result$Z2, 
                                                forward_result$A2, W1, W2, X, Y, m)
        
        updated_params <- update_paramaters(W1, b1, W2, b2, backward_result$dW1, backward_result$db1,
                                         backward_result$dW2, backward_result$db2, alpha)
        W1 <- updated_params$W1
        b1 <- updated_params$b1
        W2 <- updated_params$W2
        b2 <- updated_params$b2
        if (i %% 10 == 0) {
            #print(paste("Iteration", i, "cost:", -sum(log(forward_propagation$A2[Y + 1, seq_along(Y)])) / length(Y)))
            print(paste("Iteration", i))
            predictions <- get_predictions(forward_result$A2)
            accuracy <- get_accuracy(predictions, Y)
            print(paste("Accuracy:", accuracy))
        }
    }
}
gradient_descent2 <- function(X, Y, m, alpha, iterations) {
    params <- init_params()
    W1 <- params$W1
    b1 <- params$b1 
    W2 <- params$W2
    b2 <- params$b2
    
    for (i in 1:iterations) {
        forward_result <- forward_propagation(W1, b1, W2, b2, X)
        backward_result <- backward_propagation(forward_result$Z1, forward_result$A1, 
                                             forward_result$Z2, forward_result$A2,
                                             W1, W2, X, Y, m)
        
        updated_params <- update_paramaters(W1, b1, W2, b2,
                                         backward_result$dW1, backward_result$db1,
                                         backward_result$dW2, backward_result$db2, 
                                         alpha)
        
        W1 <- updated_params$W1
        b1 <- updated_params$b1
        W2 <- updated_params$W2
        b2 <- updated_params$b2
        
        if (i %% 10 == 0) {
            print(paste("Iteration:", i))
            predictions <- get_predictions(forward_result$A2)
            print(paste("Accuracy:", get_accuracy(predictions, Y)))
        }
    }
    
    return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
make_predictions <- function(X, W1, b1, W2, b2) {
    forward_test_result <- forward_propagation(W1, b1, W2, b2, X)
    predictions <- get_predictions(forward_test_result$A2)
    return(predictions)
}



new_params <- gradient_descent2(X, Y, m, 0.1, 250)
predictions <- make_predictions(X_test, new_params$W1, new_params$b1, new_params$W2, new_params$b2)
accuracy <- get_accuracy(predictions, Y_test)
print(paste("Accuracy on test set:", accuracy))






