% This function implements the Gauss-Helmert model.

function [x_opt, t_opt, y_opt, iter] = GaussHelmert(func, x0, t0, y0, x, P)

    it_max = 400;
    tol = 1e-6;

    xi = x0; yi = y0; ti = t0;
    u = size(t0, 1);
    s = size(y0, 1);

    v0 = x0 - x;
    objFunc = sum(v0.' * P * v0);
    factor = 1;

    for it = 1:it_max
        [f, g, A, B, C, D] = func(xi, ti, yi);
        c2 = size(C, 1);
        W = B * pinv(P) * B.';

        if any(isnan(W(:))) || any(isinf(W(:)))
            break;
        end

        W = pinv(W + (1e-12) * eye(size(W, 1))); W = W + (1e-12) * eye(size(W, 1));
        w = -f - B * (x - xi);
        M = [A.' * W * A, zeros(u, s), C.'; ...
                 zeros(s, u + s), D.'; ...
                 C, D, zeros(c2, c2)];
        b = [A.' * W * w; zeros(s, 1); -g];

        if any(isnan(M(:))) || any(isinf(M(:)))
            break;
        end

        aux = pinv(M + (1e-12) * eye(size(M, 1))) * b;
        dt = aux(1:u, :); dy = aux(u + 1:u + s, :);
        v = -inv(P) * B.' * (W * (A * dt - w));

        if norm(dt) < tol && norm(dy) < tol && norm(xi - x - v) < tol
            break;
        end

        if sum(v.' * P * v) > objFunc * factor
            break;
        else
            objFunc = sum(v.' * P * v);
        end

        xi = x + v; ti = ti + dt; yi = yi + dy;
    end

    iter = it;
    x_opt = xi; y_opt = yi; t_opt = ti;

end
