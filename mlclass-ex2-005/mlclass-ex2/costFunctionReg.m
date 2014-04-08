function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for i = 2 : size(theta)
    J += theta(i,1)^2;
end

J = lambda*J/2;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i = 1 : m
    h = sigmoid(X(i,:)*theta);
    J += -y(i, 1)*log(h) - (1 - y(i,1))*log(1-h);
    grad += (h - y(i, 1))*X(i, :)';
end

grad += lambda*[0;theta(2:size(theta),:)];

J /= m;
grad /= m;





% =============================================================

end
