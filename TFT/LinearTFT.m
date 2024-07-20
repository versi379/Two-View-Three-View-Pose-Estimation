% Description: 
% This function computes the Trifocal Tensor (TFT) from triplets of corresponding image
% points by algebraic minimization of the linear contraints given by the
% incidence relationships. The three projection matrices are also computed
% using the TFT.
%
% Input:
% p1, p2, p3: three vectors 3xN (homogeneous) or 2xN (cartesian)
%             containing image points for images 1,2, and 3
%
% Output:
% T: 3x3x3 array containing the trifocal tensor associated to 
%    the triplets of corresponding points
% P1, P2, P3: three estimated 3x4 projection matrices

function [T, P1, P2, P3] = LinearTFT(p1, p2, p3)

    N = size(p1, 2);
    A = zeros(4 * N, 27); % Matrix of the linear system on the parameters of the TFT

    if size(p1, 1) == 3
        p1 = p1(1:2, :) ./ repmat(p1(3, :), 2, 1);
        p2 = p2(1:2, :) ./ repmat(p2(3, :), 2, 1);
        p3 = p3(1:2, :) ./ repmat(p3(3, :), 2, 1);
    end

    for i = 1:N
        x1 = p1(1, i); y1 = p1(2, i);
        x2 = p2(1, i); y2 = p2(2, i);
        x3 = p3(1, i); y3 = p3(2, i);

        A(4 * (i - 1) + 1, :) = [x1, 0, -x1 * x2, 0, 0, 0, -x1 * x3, 0, x1 * x2 * x3, ...
                            y1, 0, -x2 * y1, 0, 0, 0, -x3 * y1, 0, x2 * x3 * y1, ...
                            1, 0, -x2, 0, 0, 0, -x3, 0, x2 * x3];
        A(4 * (i - 1) + 2, :) = [0, x1, -x1 * y2, 0, 0, 0, 0, -x1 * x3, x1 * x3 * y2, ...
                            0, y1, -y1 * y2, 0, 0, 0, 0, -x3 * y1, x3 * y1 * y2, ...
                            0, 1, -y2, 0, 0, 0, 0, -x3, x3 * y2];
        A(4 * (i - 1) + 3, :) = [0, 0, 0, x1, 0, -x1 * x2, -x1 * y3, 0, x1 * x2 * y3, ...
                            0, 0, 0, y1, 0, -x2 * y1, -y1 * y3, 0, x2 * y1 * y3, ...
                            0, 0, 0, 1, 0, -x2, -y3, 0, x2 * y3];
        A(4 * (i - 1) + 4, :) = [0, 0, 0, 0, x1, -x1 * y2, 0, -x1 * y3, x1 * y2 * y3, ...
                            0, 0, 0, 0, y1, -y1 * y2, 0, -y1 * y3, y1 * y2 * y3, ...
                            0, 0, 0, 0, 1, -y2, 0, -y3, y2 * y3];
    end

    [~, ~, V] = svd(A);

    t = V(:, end);
    T = reshape(t, 3, 3, 3);

    % Compute epipoles
    [~, ~, V] = svd(T(:, :, 1)); v1 = V(:, end);
    [~, ~, V] = svd(T(:, :, 2)); v2 = V(:, end);
    [~, ~, V] = svd(T(:, :, 3)); v3 = V(:, end);
    [~, ~, V] = svd([v1 v2 v3].'); epi31 = V(:, end);
    [~, ~, V] = svd(T(:, :, 1).'); v1 = V(:, end);
    [~, ~, V] = svd(T(:, :, 2).'); v2 = V(:, end);
    [~, ~, V] = svd(T(:, :, 3).'); v3 = V(:, end);
    [~, ~, V] = svd([v1 v2 v3].'); epi21 = V(:, end);

    % Matrix E (tensor linear relationship)
    E = [kron(eye(3), kron(epi31, eye(3))), -kron(eye(9), epi21)];
    [U, S, V] = svd(E); Up = U(:, 1:rank(E)); Vp = V(:, 1:rank(E)); Sp = S(1:rank(E), 1:rank(E));
    [~, ~, V] = svd(A * Up); tp = V(:, end);
    t = Up * tp;
    a = Vp * inv(Sp) * tp;

    % Compute projection matrices and trifocal tensor
    P1 = eye(3, 4);
    P2 = [reshape(a(1:9), 3, 3), epi21];
    P3 = [reshape(a(10:18), 3, 3), epi31];
    T = reshape(t, 3, 3, 3);

end
