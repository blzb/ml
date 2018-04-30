function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
a2 = sigmoid(Theta1 * [ones(m ,1), X]');
a3 = sigmoid(Theta2 * [ones(m ,1)'; a2]);
h = a3;

classes = eye(num_labels);
for i =1: m
    y_mapped(i,:) = classes(y(i), :);
end
j_partial = 0;
for i = 1: m
    for k = 1: num_labels
        j_partial = j_partial + ((-1)*y_mapped(i,k) * log(h(k, i))) - (1 - y_mapped(i,k))*log(1-h(k, i));
    end    
end
theta1_size = size(Theta1);
theta2_size = size(Theta2);
reg = 0;
for j = 1: theta1_size(1)
    for k = 2: theta1_size(2)
        reg = reg + Theta1(j,k)^2;
    end
end

for j = 1: theta2_size(1)
    for k = 2: theta2_size(2)
        reg = reg + Theta2(j,k)^2;
    end
end
reg = ((lambda)/(2*m))*reg;
J = j_partial/m + reg;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for i =1:m
    a1_i = X(i,:)';
    a1_i = [1; a1_i];
    z2_i = Theta1 * a1_i;
    a2_i = sigmoid(z2_i);
    a2_i = [1 ; a2_i];
    z3_i = Theta2 *  a2_i;
    a3_i = sigmoid(z3_i);
    h = a3_i;
    delta3_i = a3_i - y_mapped(i, :)';
    delta2_i =  (Theta2'*delta3_i).* sigmoidGradient([1; z2_i]);
    delta2_i = delta2_i(2: end);
    Delta2 = Delta2 + delta3_i * a2_i';
    Delta1 = Delta1 + delta2_i * a1_i';
end

reg1 = [zeros(theta1_size(1), 1) (lambda/m)*Theta1(:, 2:end)];
reg2 = [zeros(theta2_size(1), 1) (lambda/m)*Theta2(:, 2:end)];
Theta1_grad = (1/m) * Delta1 + reg1;
Theta2_grad = (1/m) * Delta2 + reg2;




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
