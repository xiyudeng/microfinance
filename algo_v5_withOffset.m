% micro-credit
clear all; close all; clc;
rng(1)
%% initialization
t = 5000; % number of period
% info = cell([1 t]);
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
r = 0.1; % interest rate

Rbar = 0; % Rbar
ninfo = 1; % number of information for each applicat (ninfo entries in s)
nempty = 0;

phi = realmin*ones([ninfo,1]); % control parameters
eps = realmin*ones([ninfo,1]); % offset parameters
z = [phi;eps];
% phis = [];
R_cum = zeros([t,1]);
% randR_cum = zeros([t,1]);
% ratioAs = [];

%% for each period
for i = 1:t
    % from environment

    % generate applicants number & info
%     N = randi([500,1500],1); % Nt as random number in [500, 1500]
    N = 10000; %randi([10000,20000],1);
    Nt = [Nt,N];
%     info{1,i}.Nt = N; % N applicants

    % Personal information: N x ninfo
    s = random_apc_info(N,ninfo,nempty);
%     info{1,i}.s = s;


    % calculate pie
%     info{1,i}.phi = phi;
%     phis = [phis;phi];
%     info{1,i}.gamma = gamma;
%     info{1,i}.z = z;

    s_eps = s + eps'; s_eps(isnan(s_eps)) = 0;
    theta1 = s_eps * phi(); % Nx1

%     theta0 = s * gamma(:); % Nx1

    % policy pi
%     pie = exp(theta1) ./ (exp(theta1)+exp(theta0)); 
    exp_t1 = exp(theta1);
        exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
    
    pie = exp_t1./(1+exp_t1);
        pie(pie == 0) = realmin;
%     info{1,i}.pie = pie;

    % make decision
    % random varialbe to decide if bank lend/reject applicants
    decision_varialbe = rand(N,1);
    A = (decision_varialbe < pie);
%     info{1,i}.A = A;
%     info{1,i}.numA = sum(A);
%     ratioA = info{1,i}.numA/N;
%     info{1,i}.ratioA = ratioA;
%     ratioAs = [ratioAs;ratioA];


    % calculate return & get profit
    p = returned_prob(s,ninfo);  % probability to return the money with interest c
    % random varialbe to decide if applicant return/fail
    return_varialbe = rand(N,1);
%     info{1,i}.p = p;
    R = zeros([N,1]);
    R(A == 1 & return_varialbe < p) = 1+r;
    R(A == 1 & return_varialbe >= p) = -1;
%     info{1,i}.R = R;

    % random choose action
%     randAid = randsample(1:N,sum(A));
%     randA = zeros(size(A));
%     randA(randAid) = 1;
%     randR(randA == 1 & return_varialbe < p) = 1+r;
%     randR(randA == 1 & return_varialbe >= p) = -1;
%     info{1,i}.randR = randR;

    
    % index of the accepted applications
    Aid = find(A == 1);
    
    del_pi = partial_pi_partial_z(s,z,Aid,ninfo,N);
        del_pi(del_pi == 0) = realmin;
        del_pi(del_pi == inf) = realmax;

    Rbar = sum(R_cum)/sum(Nt);
    F = ((R - Rbar).*(del_pi./pie))/N;
        Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;
    % fix the  NaN problem
%     if length(find(abs(F)>10^308))>(3*N/4)
%         F = F./10^307;
%         F = sum(F,1);
%         F = F*10^307;
%         F(isinf(F)) = realmax;
% 
%     else
%         F = sum(F,1);
%         F(isinf(F)) = realmax;
%     end
    F = sum(F,1);
        F(isinf(F)) = realmax;
    
    
    % update paras for next 
    z = z + alpha.*F';
    phi = z(1:ninfo);
    eps = z(ninfo+1:end);

    R_cum(i) = sum(R);
%     randR_cum(i) = sum(randR);
end

figure('Color','w')
plot(R_cum)%;hold on
% plot(randR_cum,'r--')
xlabel('time');
ylabel('money received back');
title(['n_{info} = ',int2str(ninfo),'; n_{empty} = ',int2str(nempty)])


% figure(2)
% plot(phis);
% xlabel('t');
% ylabel('\phi');
% title('\phi vs time')
% 
% figure(3)
% plot(ratioAs);
% xlabel('t');
% ylabel('ratioA');
% title('ratioA vs time')


% function  = probReturn()
% % g(s_it) should be in [0, 1]
% % P(eta = c) = p_it = h(s_it)
% end
%
% function phi, gamma = updateZ()
%
% end

%%
function apcs_info = random_apc_info(n_apcs,ninfo,nempty)
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
function del_pi = partial_pi_partial_z(s,z,Aid,ninfo,N)
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
    s(isnan(s)) = 0;
    p = sum(s,2) ./ (4*ninfo);
end