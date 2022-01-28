% micro-credit caseA
clear all; close all; clc;
rng(1)
%% case option
L_form = 'S';

%% initialization
t = 1000; % number of period
Nt = zeros(t,1); % list to store applicants number
alpha = 0.01; % step size / learning rate
d_alpha = alpha/t;
c = 0.1; % interest rate
e = 1;
nempty = 0;
Rbar = 0; % Rbar
ninfo = 100; % number of information for each applicat (ninfo entries in s)
% control parameters
phi = 0.1*ones([ninfo,1]);
phis = zeros(t,ninfo);
eps = 0.1*ones([ninfo,1]);
eps_arr = zeros(t,ninfo);
z = [phi;eps];
F = zeros(1,2*ninfo);
numA = zeros(t,1);

R_cum = zeros([t,1]);
R_avg = zeros([t,1]);
ratioAs = zeros(t,1);
phat = 1;
bankR_avg= zeros([t,1]);
bank_ratioAs = zeros(t,1);
s_database = [];

%% for each time step
for i = 1:t
    % follow up progress
    %     if mod(i,100) == 0
    %         disp(i);
    %     end
    waitbar(i/t)

    % generate applicants number & info
    N = 20000; %randi([10000,20000],1);
    Nt(i) = N;

    % Personal information: N x ninfo
    s = random_apc_info(N, ninfo, nempty);

    % calculate pie
    phis(i,:) = phi';
    eps_arr(i,:) = eps';

    s_phi = s.*phi';
    Q = s_phi + eps'; Q(isnan(Q)) = 0;


    % policy pi
    exp_Q = exp(mean((Q),2));
    exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

    % calculate return & get profit
    p = returned_prob(s, ninfo);  % probability to return the money with interest c
    % random varialbe to decide if bank lend/reject applicants
    decision_varialbe = rand(N,1);
    switch L_form
        case 'A'
            pie = exp_Q./(1+exp_Q);
        case 'B'
            pie = 1-(1./exp_Q);
        case 'C'
            pie = (2.*exp_Q./(1+exp_Q))-1;
        case 'S'
            bank1A = ((p - (1/(2*(c+e)))) > 0);
            bank1R = zeros([N,1]);
            bank1R(bank1A == 1 & decision_varialbe < p) = c+e;
            bank1R(bank1A == 1 & decision_varialbe >= p) = -1+e;
            bank1_ratioA = sum(bank1A)/N;
            R_cum(i) = sum(bank1R);
%             [bankR_avg(i), bank_ratioAs(i), s_database] = bank_alg(i,N,s,s_database,c,p,ninfo,phat);
    end

    if L_form ~= 'S'
        % make decisions
        A = (decision_varialbe < pie);
        ratioA = sum(A)/N;
        numA(i) = sum(A);

        % random varialbe to decide if applicant return/fail
        return_varialbe = rand(N,1);
        R = zeros([N,1]);
        R(A == 1 & return_varialbe < p) = c+e;
        R(A == 1 & return_varialbe >= p) = -1+e;
        ratioAs(i) = ratioA;
        %     bank1_ratioAs = [bank1_ratioAs;bank1_ratioA];

%         random choosing action
            randAid = randsample(1:N,sum(A));
            randA = zeros(size(A));
            randA(randAid) = 1;
            randR(randA == 1 & return_varialbe < p) = 1+c;
            randR(randA == 1 & return_varialbe >= p) = -1;

        % index of the accepted applications
        Aid = find(A == 1);
        del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form);
        del_pi(del_pi == 0) = realmin;
        del_pi(del_pi == inf) = realmax;
        pie(pie == 0) = realmin;

        Rbar = sum(R_cum)/sum(Nt);

        deltaR = (R - Rbar);
        F = mean((del_pi./pie).*deltaR,1);
        Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;

        % update paras for next
        z = z + alpha.*F';

        if L_form == 'B' || L_form == 'C'
            z(z<0) = 0;
        end

        phi = z(1:ninfo);
        eps = z(ninfo+1:end);

        %%

        R_cum(i) = sum(R);
        randR_cum(i) = sum(randR);
        %     bank1R_cum(i) = sum(bank1R);

        R_avg(i) = R_cum(i)/N;
        %     randR_avg(i) = randR_cum(i)/N;
        %     bank1R_avg(i) = bank1R_cum(i)/N;

        alpha = alpha - d_alpha;
        % alpha = alpha/sqrt(i);
    end

end
numA = cumsum(numA);

%% plots
figure('Color','w')
% subplot(3,1,1);
plot(numA,movmean(R_cum,100))%;hold on
plot(movmean(R_cum,100))
% plot(randR_avg,'r');
% plot(bank1R_avg,'g');
% plot(bankR_avg,'m');
xlabel('time');
ylabel('cumulative utility');
% legend('Gradients','Random','Standard','Standard new');
% title('rewards vs time')

if exist('randR_cum')
figure('Color','w')
plot(numA,randR_cum)%;hold on
xlabel('time');
ylabel('cumulative utility: random select');
end



figure('Color','w')
% subplot(3,1,1);
% subplot(3,1,2);
plot(phis);
xlabel('time');
ylabel('\phi');
% title('\phi vs time')

figure('Color','w')
% subplot(3,1,1);
% subplot(3,1,2);
plot(eps_arr);
xlabel('time');
ylabel('\epsilon');
% title('\phi vs time')

figure('Color','w')
% subplot(3,1,3);
plot(ratioAs)%;hold on
% plot(bank1_ratioAs,'g');
% plot(bank_ratioAs,'m');
% legend('Gradients & Random', 'Standard');
xlabel('time');
ylabel('ratioA');
% title('ratioA vs time');

%%
function apcs_info = random_apc_info(n_apcs, ninfo, nempty)
%apcs_info1 = zeros([20000,1]);
%apcs_info2 = 4*ones([20000,1]);
%apcs_info = [apcs_info1 apcs_info2]
%apcs_info = 4*ones([20000,1]);
apcs_info = randi([0 4],n_apcs,ninfo);
% apcs_info = zeros(n_apcs,ninfo);
% apcs_info(1:n_apcs,:) = 4;
%     disp(size(apcs_info));
%apcs_info = [apcs_info1 apcs_info2]
emty_idx = randi([1,numel(apcs_info(:))],nempty,1);
apcs_info(emty_idx) = NaN;
end

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


%%
function del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form)

del_pi = zeros([N, 2*ninfo]);

exp_Q = exp(Q); % Q: Nxninfo
exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

exp_plus = exp_Q + 1;
exp_sqr = exp_plus.^2;
exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;

switch L_form
    case 'A'
        par_del = exp_Q./exp_sqr;
    case 'B'
        par_del = 1./exp_Q;
    case 'C'
        par_del = 2.*exp_Q./exp_sqr;
end

del_pi(Aid,1:ninfo) = s(Aid,:).*par_del(Aid,:);
del_pi(Aid,ninfo+1:end) = par_del(Aid,:);
del_pi(~Aid,1:ninfo) = -1.*s(~Aid,:).*par_del(~Aid,:);
del_pi(~Aid,ninfo+1:end) = -1.*par_del(~Aid,:);
del_pi(del_pi==0) = realmin; del_pi(isinf(del_pi)) = realmax;
del_pi(isnan(del_pi)) = realmin;

end

%%
function p = returned_prob(s,ninfo)
% calculate the probability
p = sum(s,2) ./ (4*ninfo);
%     p = sum(s,2);
end
