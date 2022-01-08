clear all; close all; clc;

%% disclaimer
% This file is a scratch work I use to understand the algorithm

%% simulation

r = 0.01; % interest rate
Rbar = 1; % Rbar
alpha = 0.001; % alpha variable used in the gradient ascent algorithm
ninfo = 5; % number of information for each application

% control parameters
phi = 0.1*ones(ninfo,1);
gamma = 0.1*ones(ninfo,1);
z = [phi;gamma];

t_end = 1000; % intended end time (number of iteration)

R_cum = zeros(1,t_end); % initiate cumulative reward each time period

% iterate each time period
for t_idx = 1:t_end
    
    % number of applications at time period
    Nt = randi([0,50],1);
    
    % generate application informations
    s = generate_apc_info(Nt);
    
    % actual probability that the applicant will return the loan
    p = returned_prob(s);
    
    % predicted probability use to decide the acceptance
    pi = acceptance_prob(s,phi,gamma);
    
    % decide to accept or decline the applications
    a = zeros(size(pi));
    a(pi >= 0.5) = 1;
    
    % observe the profit/loss
    R = zeros(size(a));
    R(a == 1 & p > 0.5) = r;
    R(a == 1 & p <= 0.5) = -1;
    
    % update the control parameters
    theta1 = s.*phi'; % tetha when accepted
    theta0 = s.*gamma'; % tethat when declined
    exp_t1 = exp(theta1); exp_t1(exp_t1==0) = realmin;
    exp_t0 = exp(theta0); exp_t0(exp_t0==0) = realmin;
    exp_T1T0 = exp_t1 + exp_t0; exp_T1T0(exp_T1T0==0) = realmin;
    exp_T1T0_sqr = exp_T1T0.^2; exp_T1T0_sqr(exp_T1T0_sqr==0) = realmin;
    
    acctd_idx = a == 1; % index of the accepted applications
    
    % gradient of the probability
    del_pi = [zeros(size(s)),zeros(size(s))];
    del_pi(acctd_idx,1:ninfo) = ...
        ((exp_t1(acctd_idx,:).*s(acctd_idx,:).*exp_T1T0(acctd_idx,:))...
        + ((exp_t1(acctd_idx,:).^2).*s(acctd_idx,:)))...
        ./exp_T1T0_sqr(acctd_idx,:);
    del_pi(~acctd_idx,1:ninfo) = ...
        (exp_t1(~acctd_idx,:).*exp_t0(~acctd_idx,:).*s(~acctd_idx,:))...
        ./ exp_T1T0_sqr(~acctd_idx,:);
    del_pi(~acctd_idx,ninfo+1:end) = ...
        ((exp_t0(~acctd_idx,:).*s(~acctd_idx,:).*exp_T1T0(~acctd_idx,:))...
        + ((exp_t0(~acctd_idx,:).^2).*s(~acctd_idx,:)))...
        ./exp_T1T0_sqr(~acctd_idx,:);
    del_pi(acctd_idx,ninfo+1:end) = ...
        (exp_t1(acctd_idx,:).*exp_t0(acctd_idx,:).*s(acctd_idx,:))...
        ./ exp_T1T0_sqr(acctd_idx,:);
    
    F = sum((R-Rbar).*(del_pi./pi),1);
    
    z = z + alpha.*F'; % update parameters
    
    % reform the parameters
    phi = z(1:ninfo);
    gamma = z(ninfo+1:end);
    
    % cummulate the rewards
    R_cum(t_idx) = sum(R);
    
end

%% plotting the cumulative reward

figure('Color','w') % initiate the figure
plot(1:t_end,R_cum,'k') % plot the reward each time

xlabel('time')
ylabel('cumulative reward')

%% function to randomly generate the informations in s

function apcs_info = generate_apc_info(n_apcs)
% generate_apc_info() is a function to generate random application. 
%
% The output apc_info is a vector containing the quantify information of 
% the application in the following order:
%   [credit history, current income, living area, education level, existing
%   debt]
%
% The input n_apcs is an integer indicates the number of applications that
% want to be generated.

%% credit history
% The credit hisstory will be a random integer generated from a normal 
% distribution with mean 650 and standard deviation 75. Here, the higher 
% the score indicates that the applicant has a better responsibility
% toward his/her credit history thus has a higher probability that he/she
% will honor his/her responsibility to pay the loan. The random number is
% generated from a 

cr_hist = ceil(normrnd(650,75,[n_apcs,1]));

%% current income
% The current income will be a random number generated from a normal 
% distribution with mean 1 and standard deviation 0.5 (negative number will
% be set to zero) which shows the current income of the applicant.

incm = normrnd(1,0.5,[n_apcs,1]); incm(incm<0) = 0;

%% living area
% The living area will be a uniform random integer from 1 to 10 that 
% indicates the location where the applicant is currently live.

lvng_area = randi([1,10],[n_apcs,1]);

%% education level
% The education level will be a random integer from 1 to 4 generated from 
% the right hand side of the normal distribution with 1 as the mean and 
% standard deviation 1 that indicates the highest education level of the 
% applicant where:
%   1 => no education until elementary school graduate
%   2 => high school graduate 
%   3 => college graduate
%   4 => post-graduate school graduate
% Any number higher than 4 will be converted to 4

edu_lvl = ceil(abs(normrnd(1,1,[n_apcs,1]))); edu_lvl(edu_lvl > 4) = 4;

%% existing debt
% The existing debt will be a random number generated from a normal 
% distribution with mean 0.5 and standard deviation 0.25 (negative number 
% will be set to zero) which shows the current the debt that that the 
% applicant own before applying. The higher the
% current debt will likely to be the higher risk.

extng_debt = normrnd(0.5,0.25,[n_apcs,1]); extng_debt(extng_debt<0) = 0;

%% compile the information

apcs_info = [cr_hist, incm, lvng_area, edu_lvl, extng_debt];

end

%% function to calcuate the actual returned probability

function p = returned_prob(s)

% point based on credit history
crd_pt = s(:,1);
crd_pt(crd_pt <= 560) = 0;
crd_pt(crd_pt > 560 & crd_pt <= 650) = 1;
crd_pt(crd_pt > 650 & crd_pt <= 700) = 2;
crd_pt(crd_pt > 700 & crd_pt <= 750) = 3;
crd_pt(crd_pt > 750) = 4;

% point based on current income
incm_pt = s(:,2);
incm_pt(s(:,2) <= 0.25) = 0;
incm_pt(s(:,2) > 0.25 & s(:,2) <= 0.75) = 1;
incm_pt(s(:,2) > 0.75 & s(:,2) <= 1.5) = 2;
incm_pt(s(:,2) > 1.5 & s(:,2) <= 2) = 3;
incm_pt(s(:,2) > 2) = 4;

% point based on living area
area_pt = ceil(s(:,3)./3);

% point based on education level
edu_pt = s(:,4);

% point based on existing debt
debt_pt = s(:,5);
debt_pt(s(:,5) <= 0.1) = 4;
debt_pt(s(:,5) > 0.1 & s(:,5) <= 0.3) = 3;
debt_pt(s(:,5) > 0.3 & s(:,5) <= 0.6) = 2;
debt_pt(s(:,5) > 0.6 & s(:,5) <= 1) = 1;
debt_pt(s(:,5) > 1) = 0;

% calculate the probability
p = (crd_pt + incm_pt + area_pt + edu_pt + debt_pt) ./ 20;

% add appreciation point
p(crd_pt>2 & debt_pt>2) = p(crd_pt>2 & debt_pt>2) + 0.1;

% add uncertainty
p = p + normrnd(0,0.05,size(p));
p(p<0) = 0; p(p>1) = 1;

end

%% function to calculate the acceptance probability

function pi = acceptance_prob(s,phi,gamma)
% Function to calculate the probability to accept or reject an application
% given the control parameters phi and gamma

theta1 = s*phi(:);
theta0 = s*gamma(:);

pi = theta1 ./ (theta1+theta0);
pi(isnan(pi)) = 0;

end

%%



