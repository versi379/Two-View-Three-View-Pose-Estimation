function [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp)

    N = size(Corresp, 2);
    K1 = CalM(1:3, :); K2 = CalM(4:6, :); K3 = CalM(7:9, :);

    % 'remove' calibration from the tensor
    T = TransformTFT(T, K1, K2, K3, 1);

    % epipoles and essential matrix
    [~, ~, V] = svd(T(:, :, 1)); v1 = V(:, end);
    [~, ~, V] = svd(T(:, :, 2)); v2 = V(:, end);
    [~, ~, V] = svd(T(:, :, 3)); v3 = V(:, end);
    [~, ~, V] = svd([v1 v2 v3].'); epi31 = V(:, end) * sign(V(end));

    [~, ~, V] = svd(T(:, :, 1).'); v1 = V(:, end);
    [~, ~, V] = svd(T(:, :, 2).'); v2 = V(:, end);
    [~, ~, V] = svd(T(:, :, 3).'); v3 = V(:, end);
    [~, ~, V] = svd([v1 v2 v3].'); epi21 = V(:, end) * sign(V(end));

    E21 = CrossProdMatrix(epi21) * [T(:, :, 1) * epi31 T(:, :, 2) * epi31 T(:, :, 3) * epi31];
    E31 = -CrossProdMatrix(epi31) * [T(:, :, 1).' * epi21 T(:, :, 2).' * epi21 T(:, :, 3).' * epi21];

    % Find R2 and t2 from E21
    [R2, t2] = ExtractRTfromEM(E21, K1, K2, Corresp(1:2, :), Corresp(3:4, :));

    % Find R3 and t3 from E31
    [R3, t3] = ExtractRTfromEM(E31, K1, K3, Corresp(1:2, :), Corresp(5:6, :));

    % Find the norm of t3 using the image points and reconstruction from
    % images 1 and 2
    u3 = K3 * t3;
    X = Triangulate3DPoints({K1 * eye(3, 4), K2 * [R2, t2]}, Corresp(1:4, :));
    X = X(1:3, :) ./ repmat(X(4, :), 3, 1);
    X3 = K3 * R3 * X;
    lam = -sum(dot(cross([Corresp(5:6, :); ones(1, N)], X3, 1), cross([Corresp(5:6, :); ones(1, N)], repmat(u3, 1, N)), 1)) / ...
        sum(sum(cross([Corresp(5:6, :); ones(1, N)], repmat(u3, 1, N)) .^ 2));
    t3 = lam * t3;

    R_t_2 = [R2, t2]; R_t_3 = [R3, t3];

end

