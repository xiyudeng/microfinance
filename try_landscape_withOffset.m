clear all; close all; clc;
rng(1)
%% disclaimer
% This file is a scratch work I use to understand the algorithm

%% create landscape

r = 0.1; % interest rate
ninfo = 5; % number of information for each applicat (ninfo entries in s)
nempty = 1000;

% control parameters
max_inspect = 5;
[phi,eps] = ndgrid(linspace(realmin,max_inspect,100)); % gridding
R_cum = zeros(size(phi)); % initiate cumulative reward each time period

% iterate each time period
for R_idx = 1:numel(R_cum)
    
    % number of applications at time period
    Nt = 20000; %randi([10000,20000],1);
    
    % generate application informations
%     s = (abs(normrnd(1,1,[Nt,1]))); s(s > 4) = 4;
    s = randi([0 4],Nt,ninfo);
    emty_idx = randi([1,numel(s(:))],nempty,1);
    s(emty_idx) = NaN;
    
    % actual probability that the applicant will return the loan
    s_p = s; s_p(isnan(s)) = 0;
    p = sum(s_p,2) ./ (4*ninfo);
    
    % predicted probability use to decide the acceptance
    s_eps = s + eps(R_idx); s_eps(isnan(s_eps)) = 0;
    theta1 = s_eps * phi(R_idx);
    exp_t1 = exp(theta1);
        exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
    pie = exp_t1./(1+exp_t1);
    
    % decide to accept or decline the applications
    decision_varialbe = rand(Nt,1);
    A = (decision_varialbe < pie);
    
    % observe the profit/loss
    return_varialbe = rand(Nt,1);
    R = zeros([Nt,1]);
    R(A == 1 & return_varialbe < p) = 1+r;
    R(A == 1 & return_varialbe >= p) = -1;
    
    % cummulate the rewards
    R_cum(R_idx) = sum(R);
    
end

%% get maximum points

max_idx = find(R_cum==max(R_cum(:)));

%% plotting the landscape

figure('Color','w') % initiate the figure
contourf(phi,eps,R_cum,'LineStyle','none')
hold on
plot(phi(max_idx),eps(max_idx),'r*')

xlabel('\phi')
ylabel('\epsilon')
title(['n_{info} = ',int2str(ninfo),'; n_{empty} = ',int2str(nempty)])
% colormap('jet')

clr_br = colorbar;
clr_br.Label.String = 'cumulative reward';


% %% simulation
% 
% Rbar = 0; % initiate Rbar
% alpha = 1e-3; % alpha variable used in the gradient ascent algorithm
% ninfo = 1; % number of information for each application
% 
% t_end = 200; % intended end time (number of iteration)
% 
% % initiate control parameters
% phi = 0.1;
% gamma = 0.1;
% z = [phi(end);gamma(end)];
% 
% R_cum = zeros(1,t_end); % initiate cumulative reward each time period
% % Rbar = zeros(1,t_end); % initiate Rbar vector
% Nt_cum = 0;
% % iterate each time period
% for t_idx = 1:t_end
%     
%     % number of applications at time period
%     Nt = 10000;%randi([10000,20000],1);
%     Nt_cum = Nt_cum+Nt;
%     
%     % generate application informations
%     s = randi([0 4],Nt,1);
%     
%     % actual probability that the applicant will return the loan
%     p = s./4;
%     
%     % predicted probability use to decide the acceptance
%     pi_s = acceptance_prob(s,phi(end),gamma(end));
%     
%     % decide to accept or decline the applications
%     a = zeros(size(pi_s));
%     a_prob = rand([Nt,1]);
%     a(pi_s > a_prob) = 1;
%     
%     % observe the profit/loss
%     R = zeros(size(a));
%     R_prob = rand([Nt,1]);
%     R(a == 1 & p > R_prob) = 1+r;
%     R(a == 1 & p <= R_prob) = -1;
%     
% %     % update the control parameters
% %     Aid = find(a == 1);
% %     % gradient of the probability
% %     del_pi = zeros([Nt, 2*ninfo]);
% %     tmp1 = s.*phi(end); 
% %     tmp0 = s.*gamma(end); 
% %     exp_t1 = exp(tmp1);
% %     exp_t0 = exp(tmp0);
% %     exp_plus = exp_t1 + exp_t0;
% %     exp_sqr = exp_plus.^2;
% %     
% %     del_pi(Aid,1:ninfo) = ((exp_t1(Aid,:).*s(Aid,:).*exp_plus(Aid,:))...
% %         + ((exp_t1(Aid,:).^2).*s(Aid,:)))./exp_sqr(Aid,:);
% %     del_pi(~Aid,1:ninfo) = (exp_t1(~Aid,:).*exp_t0(~Aid,:).*s(~Aid,:))...
% %         ./ exp_sqr(~Aid,:);
% %     del_pi(Aid,ninfo+1:end) = (exp_t1(Aid,:).*exp_t0(Aid,:).*s(Aid,:))...
% %         ./ exp_sqr(Aid,:);
% %     del_pi(~Aid,ninfo+1:end) = ((exp_t0(~Aid,:).*s(~Aid,:).*exp_plus(~Aid,:))...
% %         + ((exp_t0(~Aid,:).^2).*s(~Aid,:)))./exp_sqr(~Aid,:);
%     
%     
%     % update the control parameters
%     theta1 = s.*phi(end); theta1(theta1==0) = realmin; % tetha when accepted
%     theta0 = s.*gamma(end); theta0(theta0==0) = realmin; % tethat when declined
%     exp_t1 = exp(theta1); 
%         exp_t1(exp_t1==0) = realmin; exp_t1(isinf(exp_t1)) = realmax;
%     exp_t0 = exp(theta0); 
%         exp_t0(exp_t0==0) = realmin; exp_t0(isinf(exp_t0)) = realmax;
%     exp_T1T0 = exp_t1 + exp_t0; 
%         exp_T1T0(exp_T1T0==0) = realmin; 
%         exp_T1T0(isinf(exp_T1T0)) = realmax;
%     exp_T1T0_sqr = exp_T1T0.^2; 
%         exp_T1T0_sqr(exp_T1T0_sqr==0) = realmin;
%         exp_T1T0_sqr(isinf(exp_T1T0_sqr)) = realmax;
%     
%     acctd_idx = find(a == 1); % index of the accepted applications
%     % gradient of the probability
%     del_pi = zeros([Nt, 2*ninfo]);
%     del_pi(acctd_idx,1:ninfo) = ...
%         ((exp_t1(acctd_idx,:).*s(acctd_idx,:).*exp_T1T0(acctd_idx,:))...
%         + ((exp_t1(acctd_idx,:).^2).*s(acctd_idx,:)))...
%         ./exp_T1T0_sqr(acctd_idx,:);
%     del_pi(~acctd_idx,1:ninfo) = ...
%         (exp_t1(~acctd_idx,:).*exp_t0(~acctd_idx,:).*s(~acctd_idx,:))...
%         ./ exp_T1T0_sqr(~acctd_idx,:);
%     del_pi(~acctd_idx,ninfo+1:end) = ...
%         ((exp_t0(~acctd_idx,:).*s(~acctd_idx,:).*exp_T1T0(~acctd_idx,:))...
%         + ((exp_t0(~acctd_idx,:).^2).*s(~acctd_idx,:)))...
%         ./exp_T1T0_sqr(~acctd_idx,:);
%     del_pi(acctd_idx,ninfo+1:end) = ...
%         (exp_t1(acctd_idx,:).*exp_t0(acctd_idx,:).*s(acctd_idx,:))...
%         ./ exp_T1T0_sqr(acctd_idx,:);
%     del_pi(del_pi==0) = realmin;
%     del_pi(isinf(del_pi)) = realmax;
% 
%     
% %     delpi0pi = del_pi./pi; %delpi0pi(isinf(delpi0pi)) = realmax;
%     
% %     avgR = mean(R);
% %         if isnan(avgR)
% %             avgR = 0;
% %         end
% %     if t_idx == 1
% %     Rbar(t_idx) = avgR;
% %     else
% %     Rbar(t_idx) = Rbar(t_idx-1) + avgR;
% %     end
% %     F = sum((R-mean(Rbar)).*(delpi0pi),1);
% %     Rbar = Rbar+alpha*(2-avgR);
% %     F = sum((R-Rbar).*(delpi0pi),1);
% %     F = sum((R-Rbar(1)).*(delpi0pi),1);
% %     F = sum((R-Rbar(1)).*(del_pi./[theta1,theta0]),1);
% %     F = sum((R-mean(R)).*(delpi0pi),1);
% %     F = sum((R-mean(R_cum)).*(delpi0pi),1);
% %     F = sum((R-(sum(R_cum)/Nt_cum)).*(delpi0pi),1);
%     F = sum((R - Rbar).*(del_pi./pi_s),1);
%         F(isinf(F)) = realmax;
%     
%     z = z + alpha.*F'; % update parameters
% %     z(isinf(z)) = realmax;
%     
%     % reform the parameters
%     phi = [phi,z(1:ninfo)];
%     gamma = [gamma,z(ninfo+1:end)];
%     
%     % cummulate the rewards
%     R_cum(t_idx) = sum(R);
%     Rbar = sum(R_cum)/Nt_cum;
%     
% end
% 
% %% plotting the steps
% 
% plot(phi,gamma,'k','LineWidth',1)
% xlim([0,max_inspect])
% ylim([0,max_inspect])

%% function to calculate the acceptance probability

% function pi = acceptance_prob(s,phi,gamma)
% % Function to calculate the probability to accept or reject an application
% % given the control parameters phi and gamma
% 
% theta1 = s*phi(:);
% theta0 = s*gamma(:);
% exp_t1 = exp(theta1); 
% exp_t0 = exp(theta0); 
% 
% pi = exp_t1 ./ (exp_t1+exp_t0);
% pi(isnan(pi)) = realmin;
% 
% end

%%



