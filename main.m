% micro-credit
clear all; close all; clc;

%% initialization
t = 100; % number of period
info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.001; % interest rate
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
    p = returned_prob(s);  % probability to return the money with interest r
    info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & p > 0.5) = c;
    R(A == 1 & p <= 0.5) = -1;
    info{1,i}.R = R;

    tmp1 = s.*phi'; 
    tmp0 = s.*gamma'; 
    exp_t1 = exp(tmp1); exp_t1(exp_t1==0) = realmin;
    exp_t0 = exp(tmp0); exp_t0(exp_t0==0) = realmin;
    
    exp_plus = exp_t1 + exp_t0; exp_plus(exp_plus==0) = realmin;
    exp_sqr = exp_plus.^2; exp_sqr(exp_sqr==0) = realmin;
    
    % index of the accepted applications
    Aid = find(A == 1); 
    
    % gradient of the probability
    del_pi = zeros([N, 2*ninfo]);
    del_pi(Aid,1:ninfo) = ...
        ((exp_t1(Aid,:).*s(Aid,:).*exp_plus(Aid,:))...
        + ((exp_t1(Aid,:).^2).*s(Aid,:)))...
        ./exp_sqr(Aid,:);
    del_pi(~Aid,1:ninfo) = ...
        (exp_t1(~Aid,:).*exp_t0(~Aid,:).*s(~Aid,:))...
        ./ exp_sqr(~Aid,:);

    del_pi(Aid,ninfo+1:end) = ...
        (exp_t1(Aid,:).*exp_t0(Aid,:).*s(Aid,:))...
        ./ exp_sqr(Aid,:);
    del_pi(~Aid,ninfo+1:end) = ...
        ((exp_t0(~Aid,:).*s(~Aid,:).*exp_plus(~Aid,:))...
        + ((exp_t0(~Aid,:).^2).*s(~Aid,:)))...
        ./exp_sqr(~Aid,:);
    
    
    F = sum((R - Rbar).*(del_pi./pi),1);
    
    
    % update paras for next iter
    z = z + alpha.*F';
    phi = z(1:ninfo);
    gamma = z(ninfo+1:end);
    Rbar = Rbar + sum(R)/(N*i);

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

apcs_info = [s1, s2, s3, s3, s5];
end


%%
function p = returned_prob(s)

% calculate the probability
p = sum(s,2) ./ 20;

% add appreciation point
p(s(:,1)>2 & s(:,5)>2) = p(s(:,1)>2 & s(:,5)>2) + 0.1;

% add uncertainty
p = p + normrnd(0,0.05,size(p));
p(p<0) = 0; p(p>1) = 1;

end
