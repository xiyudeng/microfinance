% micro-credit
clear all; close all; clc;

%% initialization
t = 10; % number of period
info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.001; % interest rate
% eta = % -1 or c
% p = % probability to return the money with interest r
% pie = % policy pi

Rbar = 1; % Rbar
ninfo = 5; % number of information for each applicat (ninfo entries in s)

% control parameters
phi = 0.1*ones(ninfo,1);
gamma = 0.1*ones(ninfo,1);
z = [phi;gamma];


%% 
for i = 1:t
    % from environment

    % generate applicants number & info
    N = randi([500,1500],1); % Nt as random number in [500, 1500]
    Nt = [Nt,N];
    info{1,i}.Nt = N; % N applicants
    % Personal information: N x ninfo
    s = generate_apc_info(N);
    info{1,i}.s = s;


    % calculate pie
    info{1,i}.phi = phi;
    info{1,i}.gamma = gamma;
    info{1,i}.z = z;
    
    theta1 = s * phi(:); % Nx1
    theta0 = s * gamma(:); % Nx1

    pie = theta1 ./ (theta1+theta0);
    info{1,i}.pie = pie;
    
    % make decision
    A = zeros([N,1]);
    thre = 0.5; % threshold for lending
    A(pie >= 0.5) = 1;
    info{1,i}.A = A;
    info{1,i}.numA = sum(A);
    info{1,i}.ratioA = sum(A,1)/N;

    % calculate return & get profit
    p = returned_prob(s);
    info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & p > 0.5) = c;
    R(A == 1 & p <= 0.5) = -1;
    info{1,i}.R = R;

%     phi, gamma = updateZ()
end

% function  = probReturn()
% % g(s_it) should be in [0, 1]
% % P(eta = c) = p_it = h(s_it)
% end
% 
% function phi, gamma = updateZ()
% 
% end


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
