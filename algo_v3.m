% micro-credit
clear all; close all; clc;

%% initialization
t = 1000; % number of period
Nt = []; % list to store applicants number
alpha = 0.001; % step size / learning rate
c = 0.1; % interest rate
nempty = 0;
Rbar = 0; % Rbar
ninfo = 2; % number of information for each applicat (ninfo entries in s)
% control parameters
phi = 0.1*ones([ninfo,1]);
phis = [];
eps_arr = [];
eps = 0.1*ones([ninfo,1]);
z = [phi;eps];

R_cum = zeros([t,1]);
randR_cum = zeros([t,1]);
R_avg = zeros([t,1]);
bankR_avg = zeros([t,1]);
randR_avg = zeros([t,1]);
ratioAs = [];
bank1_ratioAs = [];
fbank_ratioAs = zeros([t,1]);
s_database = [];
phat = 1;


%% for each time step
for i = 1:t
    % from environment

    % generate applicants number & info
    N = 20000; %randi([10000,20000],1);
    Nt = [Nt,N];

    % Personal information: N x ninfo
    s = random_apc_info(N, ninfo, nempty);
    % calculate pie
    phis = [phis;phi];
    eps_arr = [eps_arr;eps];
    s_phi = s + phi'; s_phi(isnan(s_phi)) = 0;

    theta1 = s_phi * eps(:); % Nx1

    % policy pi
    exp_t1 = exp(theta1);
    exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;

    pie = exp_t1./(1+exp_t1); % problem here

    % make decision
    % random varialbe to decide if bank lend/reject applicants
    decision_varialbe = rand(N,1);
    A = (decision_varialbe < pie);
    ratioA = sum(A)/N;



    % calculate return & get profit
    p = returned_prob(s, ninfo);  % probability to return the money with interest c
    % random varialbe to decide if applicant return/fail
    return_varialbe = rand(N,1);
    R = zeros([N,1]);
    R(A == 1 & return_varialbe < p) = 1+c;
    R(A == 1 & return_varialbe >= p) = -1;

    % bank1 choosing strategy
    bank1A = ((p - (1/(1+c))) > 0);
    bank1R = zeros([N,1]);
    bank1R(bank1A == 1 & return_varialbe < p) = 1+c;
    bank1R(bank1A == 1 & return_varialbe >= p) = -1;

    bank1_ratioA = sum(bank1A)/N;
    ratioAs = [ratioAs;ratioA];
    bank1_ratioAs = [bank1_ratioAs;bank1_ratioA];

    % random choosing action
    randAid = randsample([1:N],sum(A));
    randA = zeros(size(A));
    randA(randAid) = 1;
    randR(randA == 1 & return_varialbe < p) = 1+c;
    randR(randA == 1 & return_varialbe >= p) = -1;

    [bankR_avg(i), bank_ratioAs(i), s_database] = bank_alg(i,N,s,s_database,c,p,ninfo,phat);



    % index of the accepted applications
    Aid = find(A == 1);
    del_pi = partial_pi_partial_z(z,s,phi,Aid,ninfo,N);
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

    % update paras for next
    z = z + alpha.*F';
    phi = z(1:ninfo);
    eps = z(ninfo+1:end);

    phi(phi>1.2) = 1.2;

    %%

    R_cum(i) = sum(R);
    randR_cum(i) = sum(randR);
    bank1R_cum(i) = sum(bank1R);

    R_avg(i) = R_cum(i)/N;
    randR_avg(i) = randR_cum(i)/N;
    bank1R_avg(i) = bank1R_cum(i)/N;
    


end

%% plots
figure(1)
subplot(3,1,1);
plot(R_avg);hold on
plot(randR_avg,'r');
plot(bank1R_avg,'g');
plot(bankR_avg,'m');
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
plot(bank1_ratioAs,'g');
plot(bank_ratioAs,'m');
legend('Gradients & Random', 'Standard');
xlabel('t');
ylabel('ratioA');
title('ratioA vs time');
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
%disp(size(n_apcs));
%apcs_info1 = zeros([20000,1]);
%apcs_info2 = 4*ones([20000,1]);
%apcs_info = [apcs_info1 apcs_info2]
%apcs_info = 4*ones([20000,1]);
apcs_info = randi([0 4],n_apcs,ninfo);
%     disp(size(apcs_info));
%apcs_info = [apcs_info1 apcs_info2]
emty_idx = randi([1,numel(apcs_info(:))],nempty,1);
apcs_info(emty_idx) = NaN;
end

%% compute partial pi(probability) over partial z
% problem only positive
function [R_avg, ratioA, s_database] =  bank_alg(i,N,s,s_database,c,p,ninfo, phat)
%% update the s_database and initial new entry
if i == 1
    % s_database structure
    % col 1:ninfo: s infomation
    % col ninfo+1: prob = #return / # approve
    % col ninfo+2: # return
    % col ninfo+3: # approve
    l = length(unique(s,'rows'));
    s_database = [unique(s,'rows'), phat * ones(l,1), zeros(l,1), zeros(l,1)]; %initial all prob as phat
else % i not 1
    diff_s = setdiff(unique(s,'rows'),s_database(:,1:ninfo),'rows');
    % new entry: not all entries in s are in s_database
    if (~isempty(diff_s))
        s_database = [s_database; [diff_s, zeros(size(diff_s,1),1), zeros(size(diff_s,1),1),phat*ones(size(diff_s,1),1)]]; % set prob as initial phat
    end
end

%% get the action A
% if prob >= 1/(1+c) : approve
% if prob < 1/(1+c) : approve according to prob


[~, idx_s] = ismember(s,s_database(:,1:ninfo),'rows');
%         idx_s = find(s == s_database(:,1:ninfo));
p_databse = s_database(idx_s,ninfo+1);

A = zeros(N,1); % initial A list as 0
x = rand; % a random number [0,1]
A(p_databse >= x) = 1; % apply 1 according to prob
A(p_databse >= (1/(1+c))) = 1; % case: prob >= 1/(1+c)

%% calculate the return
p_return = p;

return_varialbe = rand(N,1);
R = zeros([N,1]);
R(A == 1 & return_varialbe < p_return) = 1+c;
R(A == 1 & return_varialbe >= p_return) = -1;

R_cum = sum(R);
R_avg = R_cum/N;
ratioA = sum(A)/N;

%% update the s_database with new prob
[~,~,ix] = unique(s,'rows');
approve_count = accumarray(ix,1); % count the number of approve
approve_idx = find(ismember(s_database(:,1:ninfo),unique(s,'rows'),'rows')==1);

%         return_count = zeros(length(s_database),1);
s_return = s(R == 1+c,:);
[~,~,ix] = unique(s_return,'rows');
return_count = accumarray(ix,1); % count the number of approve
return_idx = find(ismember(s_database(:,1:ninfo),s_return,'rows')==1);
%         s_return = [s_return;return_count]; % #col:ninfo+1

s_database(return_idx,ninfo+2) = s_database(return_idx,ninfo+2)+return_count;
s_database(approve_idx,ninfo+3) = s_database(approve_idx,ninfo+3)+approve_count;

% update prob
s_database(:,ninfo+1) = s_database(:,ninfo+2)./s_database(:,ninfo+3);
end


function del_pi = partial_pi_partial_z(z,s,phi,Aid,ninfo,N)
phi = z(1:ninfo);
eps = z(ninfo+1:end);

del_pi = zeros([N, 2*ninfo]);

s_phi = s+phi'; s_phi(isnan(s_phi)) = 0;
tmp1 = s_phi.*eps';
%     tmp0 = s.*gamma';
exp_t1 = exp(tmp1);
exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
%     exp_t0 = exp(tmp0);
exp_t0 = 1;
exp_plus = exp_t1 + exp_t0;
exp_sqr = exp_plus.^2;
exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;

del_pi(Aid,1:ninfo) = ((exp_t1(Aid,:).*eps'.*exp_plus(Aid,:))...
    + ((exp_t1(Aid,:).^2).*eps'))./exp_sqr(Aid,:);
del_pi(Aid,ninfo+1:end) = ((exp_t1(Aid,:).*s_phi(Aid,:).*exp_plus(Aid,:))...
    + ((exp_t1(Aid,:).^2).*s_phi(Aid,:)))./exp_sqr(Aid,:);
del_pi(~Aid,1:ninfo) = (exp_t1(~Aid,:).*eps')...
    ./ exp_sqr(~Aid,:);
del_pi(~Aid,ninfo+1:end) = (exp_t1(~Aid,:).*s_phi(~Aid,:))./exp_sqr(~Aid,:);

del_pi(del_pi==0) = realmin; del_pi(isinf(del_pi)) = realmax;
end

%%
function p = returned_prob(s,ninfo)
% calculate the probability
p = sum(s,2) ./ (4*ninfo);
%     p = sum(s,2);
end
