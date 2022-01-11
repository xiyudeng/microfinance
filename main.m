% micro-credit
clear all; close all; clc;

%% initialization
t = 200; % number of period
info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.1; % interest rate
% eta = % -1 or c

Rbar = 0; % Rbar
ninfo = 5; % number of information for each applicat (ninfo entries in s)

% control parameters
phi = 0.1*ones([ninfo,1]);
gamma = 0.1*ones([ninfo,1]);
z = [phi;gamma];
R_cum = zeros([t,1]);

%% for each period
for i = 1:t
    % from environment

    % generate applicants number & info
    N = randi([500,1500],1); % Nt as random number in [500, 1500]
    Nt = [Nt,N];
    info{1,i}.Nt = N; % N applicants

    % Personal information: N x ninfo
    s = random_apc_info(N);
    info{1,i}.s = s;


    % calculate pie
    info{1,i}.phi = phi;
    info{1,i}.gamma = gamma;
    info{1,i}.z = z;

    theta1 = s * phi(:); % Nx1
    theta0 = s * gamma(:); % Nx1

    % policy pi
    pie = exp(theta1) ./ (exp(theta1)+exp(theta0)); 
    info{1,i}.pie = pie;

    % make decision
    % random varialbe to decide if bank lend/reject applicants
    decision_varialbe = rand(N,1);
    A = (decision_varialbe < pie);
    info{1,i}.A = A;
    info{1,i}.numA = sum(A);
    info{1,i}.ratioA = info{1,i}.numA/N;

    % calculate return & get profit
    p = returned_prob(s);  % probability to return the money with interest c
    % random varialbe to decide if applicant return/fail
    return_varialbe = rand(N,1);
    info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & return_varialbe < p) = c;
    R(A == 1 & return_varialbe >= p) = -1;
    info{1,i}.R = R;
    
    % index of the accepted applications
    Aid = find(A == 1); 
    del_pi = partial_pi_partial_z(s,z,Aid,ninfo,N);
    
    F = sum((R - Rbar).*(del_pi./pi),1);
    
    
    % update paras for next iter
    z = z + alpha.*F';
    phi = z(1:ninfo);
    gamma = z(ninfo+1:end);
    Rbar = sum(R_cum)/sum(Nt);

    R_cum(i) = sum(R);
end

plot(R_cum);

% function  = probReturn()
% % g(s_it) should be in [0, 1]
% % P(eta = c) = p_it = h(s_it)
% end
%
% function phi, gamma = updateZ()
%
% end

%%
function apcs_info = random_apc_info(n_apcs)
    % apcs_info: n_apcs * ninfo
    % credit history s1: 1(bad) -> 4(good)
    s1 = randi([0 4],n_apcs,1); 
    % current income s2: 1(bad) -> 4(good)
    s2 = randi([0 4],n_apcs,1); 
    % living area s3: 1(bad) -> 4(good)
    s3 = randi([0 4],n_apcs,1); 
    % education level s4: 1(bad) -> 4(good)
    s4 = randi([0 4],n_apcs,1); 
    % existing debt s5: 1(have debt) -> 4(no debt)
    s5 = randi([0 4],n_apcs,1); 

    apcs_info = [s1, s2, s3, s4, s5];
end

%% compute partial pi(probability) over partial z
function del_pi = partial_pi_partial_z(s,z,Aid,ninfo,N)
    phi = z(1:ninfo);
    gamma = z(ninfo+1:end);
    del_pi = zeros([N, 2*ninfo]);
    
    tmp1 = s.*phi'; 
    tmp0 = s.*gamma'; 
    exp_t1 = exp(tmp1);
    exp_t0 = exp(tmp0);
    exp_plus = exp_t1 + exp_t0;
    exp_sqr = exp_plus.^2;
    
    del_pi(Aid,1:ninfo) = ((exp_t1(Aid,:).*s(Aid,:).*exp_plus(Aid,:))...
        + ((exp_t1(Aid,:).^2).*s(Aid,:)))./exp_sqr(Aid,:);
    del_pi(~Aid,1:ninfo) = (exp_t1(~Aid,:).*exp_t0(~Aid,:).*s(~Aid,:))...
        ./ exp_sqr(~Aid,:);
    del_pi(Aid,ninfo+1:end) = (exp_t1(Aid,:).*exp_t0(Aid,:).*s(Aid,:))...
        ./ exp_sqr(Aid,:);
    del_pi(~Aid,ninfo+1:end) = ((exp_t0(~Aid,:).*s(~Aid,:).*exp_plus(~Aid,:))...
        + ((exp_t0(~Aid,:).^2).*s(~Aid,:)))./exp_sqr(~Aid,:);
end

%%
function p = returned_prob(s)
    % calculate the probability
    p = sum(s,2) ./ 20;
end
