% micro-credit
clear all; close all; clc;

%% initialization
t = 1000; % number of period
info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.1; % interest rate

Rbar = 0; % Rbar
ninfo = 1; % number of information for each applicat (ninfo entries in s)

% control parameters
phi = 0.1*ones([ninfo,1]);
phis = [];

R_cum = zeros([t,1]);
randR_cum = zeros([t,1]);
R_avg = zeros([t,1]);
randR_avg = zeros([t,1]);
ratioAs = [];
Rbars = Rbar;
delpie = [];
Ftests = [];
Fs = [];

%% for each time step
for i = 1:t
    % from environment

    % generate applicants number & info
    N = 20000; %randi([10000,20000],1);
    Nt = [Nt,N];
    info{1,i}.Nt = N; % N applicants

    % Personal information: N x ninfo
    s = random_apc_info(N);
    info{1,i}.s = s;


    % calculate pie
    info{1,i}.phi = phi;
    phis = [phis;phi];

    theta1 = s * phi(:); % Nx1

%     theta0 = s * gamma(:); % Nx1

    % policy pi
%     pie = exp(theta1) ./ (exp(theta1)+exp(theta0)); 
    exp_t1 = exp(theta1);
        exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
    
    pie = exp_t1./(1+exp_t1); % problem here
    info{1,i}.pie = pie;

    % make decision
    % random varialbe to decide if bank lend/reject applicants
    decision_varialbe = rand(N,1);
    A = (decision_varialbe < pie);
    info{1,i}.A = A;
    info{1,i}.numA = sum(A);
    ratioA = info{1,i}.numA/N;
    info{1,i}.ratioA = ratioA;
    ratioAs = [ratioAs;ratioA];


    % calculate return & get profit
    p = returned_prob(s);  % probability to return the money with interest c
    % random varialbe to decide if applicant return/fail
    return_varialbe = rand(N,1);
    info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & return_varialbe < p) = 1+c;
    R(A == 1 & return_varialbe >= p) = -1;
    info{1,i}.R = R;

    A_1 = (p > (1/(1+c));

    % random choose action
    randAid = randsample([1:N],sum(A));
    randA = zeros(size(A));
    randA(randAid) = 1;
    randR(randA == 1 & return_varialbe < p) = 1+c;
    randR(randA == 1 & return_varialbe >= p) = -1;
    info{1,i}.randR = randR;

    
    % index of the accepted applications
    Aid = find(A == 1); 
    del_pi = partial_pi_partial_z(s,phi,Aid,ninfo,N);
  
%     del_pi(del_pi == 0) = realmin;
%     del_pi(del_pi == inf) = realmax;
%     pie(pie == 0) = realmin;

    Rbar = sum(R_cum)/sum(Nt);
%     Rbar = sum(R_cum)/i;

    
    F = ((R - Rbar).*del_pi./pie)/N;
%     F = (mean(R - Rbar) * mean(del_pi./pie/N));   
%     F = mean(F);

        Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;
    
    % fix the  NaN problem
    if length(find(abs(F)>10^308))>(3*N/4)
        F = F./10^307;
        F = sum(F,1);
        F = F*10^307;
        F(isinf(F)) = realmax;

    else
        F = sum(F,1);
        F(isinf(F)) = realmax;
    end
    
    
    % update paras for next 
    phi = phi + alpha.*F';
    if phi > 1
        phi = 1;
    end

    R_cum(i) = sum(R);
    randR_cum(i) = sum(randR);
    
    R_avg(i) = R_cum(i)/N;
    randR_avg(i) = randR_cum(i)/N;
end

%% plots
figure(1)
subplot(3,1,1);
plot(R_avg);hold on
plot(randR_avg,'r')
xlabel('t');
ylabel('cumR');
title('rewards vs time')


% figure(2)
% subplot(3,1,1);
subplot(3,1,2);
plot(phis);
xlabel('t');
ylabel('\phi');
title('\phi vs time')

% figure(3)
subplot(3,1,3);
plot(ratioAs);
xlabel('t');
ylabel('ratioA');
title('ratioA vs time')

%%
function apcs_info = random_apc_info(n_apcs)
%     % apcs_info: n_apcs * ninfo
%     % credit history s1: 1(bad) -> 4(good)
%     s1 = randi([0 4],n_apcs,1); 
%     % current income s2: 1(bad) -> 4(good)
%     s2 = randi([0 4],n_apcs,1); 
%     % living area s3: 1(bad) -> 4(good)
%     s3 = randi([0 4],n_apcs,1); 
%     % education level s4: 1(bad) -> 4(good)
%     s4 = randi([0 4],n_apcs,1); 
%     % existing debt s5: 1(have debt) -> 4(no debt)
%     s5 = randi([0 4],n_apcs,1); 

    apcs_info = randi([0 4],n_apcs,1);
end

%% compute partial pi(probability) over partial z
% problem only positive
function del_pi = partial_pi_partial_z(s,phi,Aid,ninfo,N)
    del_pi = zeros([N, ninfo]);
    
    tmp1 = s.*phi'; 
    exp_t1 = exp(tmp1);
        exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
    exp_t0 = 1;
    exp_plus = exp_t1 + exp_t0;
    exp_sqr = exp_plus.^2;
        exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;
    
    del_pi(Aid,:) = (exp_t1(Aid,:).*s(Aid,:).*exp_plus(Aid,:) ...
        + (exp_t1(Aid,:).^2).*s(Aid,:))./exp_sqr(Aid,:);

    del_pi(~Aid,:) = (exp_t1(~Aid,:).*s(~Aid,:))...
        ./ exp_sqr(~Aid,:);

%     del_pi(del_pi==0) = realmin; 
%     del_pi(isinf(del_pi)) = realmax;
end

%%
function p = returned_prob(s)
    % calculate the probability
    p = sum(s,2) ./ 4;
%     p = sum(s,2);
end
