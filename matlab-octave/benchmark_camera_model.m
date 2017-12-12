close all;
clear;

% Define undistorted x and y (this is the projection of an image space point to
% the virtual image plane).
syms xu yu;

% Compute radius.
ru = sqrt(xu^2 + yu^2);

% For the fisheye case (ru > eps):
theta_by_r = atan2(ru, 1) / ru;
xf = theta_by_r * xu;
yf = theta_by_r * yu;

% For the non-fisheye case (used as a special case for ru <= eps to avoid
% division by zero):
% xf = xu;
% yf = yu;

% Define distortion parameters.
syms k1 k2 p1 p2 k3 k4 sx1 sy1;

% Compute distorted x and y (this is what would be transformed to pixel
% coordinates using fx, fy, cx, cy).
x2 = xf^2;
xy = xf * yf;
y2 = yf^2;
r2 = x2 + y2;
r4 = r2^2;
r6 = r4 * r2;
r8 = r6 * r2;

radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
dx = 2 * p1 * xy + p2 * (r2 + 2 * x2) + sx1 * r2;
dy = 2 * p2 * xy + p1 * (r2 + 2 * y2) + sy1 * r2;

xd = xf + radial * xf + dx;
yd = yf + radial * yf + dy;

% Compute Jacobian of [xd; yd] wrt. [xu; yu]:
% NOTE: The simplify() here did not seem to produce a good result in octave.
jac_coordinates = jacobian([xd; yd], [xu; yu]);
jac_coordinates_simple = simplify(jac_coordinates, 'Steps', 100);
display('Jacobian of [xd; yd] wrt. [xu; yu]:');
jac_coordinates_simple

% Compute Jacobian of [xd; yd] wrt. [k1; k2; p1; p2; k3; k4; sx1; sy1]:
jac_intrinsics = jacobian([xd; yd], [k1; k2; p1; p2; k3; k4; sx1; sy1]);
jac_intrinsics_simple = simplify(jac_intrinsics, 'Steps', 100);
display('Jacobian of [xd; yd] wrt. [k1; k2; p1; p2; k3; k4; sx1; sy1]:');
jac_intrinsics_simple
