clear all; close all; clc;
% rng(1)
%% disclaimer
% This file is a scratch work I use to understand the algorithm

%% simulation

r = 0.01; % interest rate
ninfo = 1; % number of information for each application

% control parameters
[phi,gamma] = ndgrid(linspace(0.0001,10,100)); % gridding for phi and gamma
R_cum = zeros(size(phi)); % initiate cumulative reward each time period

% iterate each time period
for R_idx = 1:numel(R_cum)
    
    % number of applications at time period
    Nt = 20000; %randi([10000,20000],1);
    
    % generate application informations
    s = (abs(normrnd(1,1,[Nt,1]))); s(s > 4) = 4;
    
    % actual probability that the applicant will return the loan
    p = s./4;
    
    % predicted probability use to decide the acceptance
    pi_s = acceptance_prob(s,phi(R_idx),gamma(R_idx));
    
    % decide to accept or decline the applications
    a = zeros(size(pi_s));
    a_prob = rand([Nt,1]);
    a(pi_s > a_prob) = 1;
    
    % observe the profit/loss
    R = zeros(size(a));
    R_prob = rand([Nt,1]);
    R(a == 1 & p > R_prob) = 1+r;
    R(a == 1 & p <= R_prob) = -1;
    
    % cummulate the rewards
    R_cum(R_idx) = sum(R);
    
end

%% plotting the cumulative reward

figure('Color','w') % initiate the figure
contourf(phi,gamma,R_cum,'LineStyle','none')

xlabel('\phi')
ylabel('\gamma')
% colormap('jet')

clr_br = colorbar;
clr_br.Label.String = 'cumulative reward';

%% function to calculate the acceptance probability

function pi = acceptance_prob(s,phi,gamma)
% Function to calculate the probability to accept or reject an application
% given the control parameters phi and gamma

theta1 = s*phi(:);
theta0 = s*gamma(:);
exp_t1 = exp(theta1);
exp_t0 = exp(theta0);

pi = exp_t1 ./ (exp_t1+exp_t0);

end

%%



