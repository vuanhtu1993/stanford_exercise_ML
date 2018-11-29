function g = sigmoidGradient(z)
% ======== ??o hàm gradient =======
% =================================
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

tmp = 1.0 ./ (1.0 + exp(-z));
g   = tmp.*(1-tmp);












% =============================================================




end
