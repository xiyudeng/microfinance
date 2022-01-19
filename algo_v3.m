% micro-credit
clear all; close all; clc;

%% initialization
t = 1000; % number of period
info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.1; % interest rate
nempty = 0;
Rbar = 0; % Rbar
ninfo = 2; % number of information for each applicat (ninfo entries in s)
% control parameters
phi = 0.1*ones([ninfo,1]);
phis = [];
eps = 0.1*ones([ninfo,1]);
z = [phi;eps];

R_cum = zeros([t,1]);
randR_cum = zeros([t,1]);
R_avg = zeros([t,1]);
randR_avg = zeros([t,1]);
ratioAs = [];
bank_ratioAs = [];


%% for each time step
for i = 1:t
    % from environment

    % generate applicants number & info
    N = 20000; %randi([10000,20000],1);
    Nt = [Nt,N];
    info{1,i}.Nt = N; % N applicants

    % Personal information: N x ninfo
    s = random_apc_info(N, ninfo, nempty);
    info{1,i}.s = s;
    % calculate pie
    info{1,i}.phi = phi;
    phis = [phis;phi];
    s_eps = s + eps'; s_eps(isnan(s_eps)) = 0;

    theta1 = s_eps * phi(:); % Nx1

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


    % calculate return & get profit
    p = returned_prob(s, ninfo);  % probability to return the money with interest c
    % random varialbe to decide if applicant return/fail
    return_varialbe = rand(N,1);
    info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & return_varialbe < p) = 1+c;
    R(A == 1 & return_varialbe >= p) = -1;
    info{1,i}.R = R;

    % bank choosing strategy
    bankA = ((p - (1/(1+c))) > 0);
    bankR = zeros([N,1]);
    bankR(bankA == 1 & return_varialbe < p) = 1+c;
    bankR(bankA == 1 & return_varialbe >= p) = -1;
    info{1,i}.bankR = bankR;
    bank_ratioA = sum(bankA)/N;
    ratioAs = [ratioAs;ratioA];
    bank_ratioAs = [bank_ratioAs;bank_ratioA];

    % random choosing action
    randAid = randsample([1:N],sum(A));
    randA = zeros(size(A));
    randA(randAid) = 1;
    randR(randA == 1 & return_varialbe < p) = 1+c;
    randR(randA == 1 & return_varialbe >= p) = -1;
    info{1,i}.randR = randR;

    
    % index of the accepted applications
    Aid = find(A == 1); 
    del_pi = partial_pi_partial_z(z,s,phi,Aid,ninfo,N);
  disp(size(del_pi));
%     del_pi(del_pi == 0) = realmin;
%     del_pi(del_pi == inf) = realmax;
%     pie(pie == 0) = realmin;

    Rbar = sum(R_cum)/sum(Nt);

    
    F = ((R - Rbar).*del_pi./pie)/N;
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
    
    disp(size(F));
    % update paras for next
    z = z + alpha.*F';
    phi = z(1:ninfo);
    eps = z(ninfo+1:end);
    
    phi(phi>1.2) = 1.2;

    R_cum(i) = sum(R);
    randR_cum(i) = sum(randR);
    bankR_cum(i) = sum(bankR);
    
    R_avg(i) = R_cum(i)/N;
    randR_avg(i) = randR_cum(i)/N;
    bankR_avg(i) = bankR_cum(i)/N;


end

%% plots
figure(1)
subplot(3,1,1);
plot(R_avg);hold on
plot(randR_avg,'r');
plot(bankR_avg,'g');
xlabel('t');
ylabel('cumR');
legend('Gradients','Random','Standard');
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
plot(ratioAs);hold on
plot(bank_ratioAs,'g');
legend('Gradients & Random', 'Standard');
xlabel('t');
ylabel('ratioA');
title('ratioA vs time')

%%
function apcs_info = random_apc_info(n_apcs, ninfo, nempty)
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

    apcs_info = randi([0 4],n_apcs,ninfo);
    emty_idx = randi([1,numel(apcs_info(:))],nempty,1);
    apcs_info(emty_idx) = NaN;
end

%% compute partial pi(probability) over partial z
% problem only positive
function del_pi = partial_pi_partial_z(z,s,phi,Aid,ninfo,N)
     phi = z(1:ninfo);
    eps = z(ninfo+1:end);

    del_pi = zeros([N, 2*ninfo]);
    
    s_eps = s+eps'; s_eps(isnan(s_eps)) = 0; 
    tmp1 = s_eps.*phi'; 
%     tmp0 = s.*gamma'; 
    exp_t1 = exp(tmp1);
        exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
%     exp_t0 = exp(tmp0);
    exp_t0 = 1;
    exp_plus = exp_t1 + exp_t0;
    exp_sqr = exp_plus.^2;
        exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;
    
    del_pi(Aid,1:ninfo) = ((exp_t1(Aid,:).*s_eps(Aid,:).*exp_plus(Aid,:))...
        + ((exp_t1(Aid,:).^2).*s_eps(Aid,:)))./exp_sqr(Aid,:);
    del_pi(Aid,ninfo+1:end) = ((exp_t1(Aid,:).*phi'.*exp_plus(Aid,:))...
        + ((exp_t1(Aid,:).^2).*phi'))./exp_sqr(Aid,:);
    del_pi(~Aid,1:ninfo) = (exp_t1(~Aid,:).*s_eps(~Aid,:))...
        ./ exp_sqr(~Aid,:);
    del_pi(~Aid,ninfo+1:end) = (exp_t1(~Aid,:).*phi')./exp_sqr(~Aid,:);
    
        del_pi(del_pi==0) = realmin; del_pi(isinf(del_pi)) = realmax;
end

%%
function p = returned_prob(s,ninfo)
    % calculate the probability
    p = sum(s,2) ./ (4*ninfo);
%     p = sum(s,2);
end
