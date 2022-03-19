function [R_cum,acceptance,default,parameters] = ...
    MFI_algo(case_option,MS_parameters,gradient_parameters,rndm)

if ~rndm
    rng(1)
end
%% case option

% reward rule
c = case_option.c; % interest rate
e = case_option.e; % encouragement

% information parameters
ninfo = case_option.ninfo; % number of information for each applicat (ninfo entries in s)
nempty_prcnt = case_option.nempty; % percentage of empty entries

% number of period
t = case_option.t;

%% MS parameters

nPartcl = MS_parameters.nPartcl;            % population size
nkeep = MS_parameters.nkeep;              % number to keep
MSO_itr = MS_parameters.itr;           % multi start iterations

%% gradient optimization parameters

% form of pi
L_form = gradient_parameters.L_form;
k = gradient_parameters.k;

% step size constant
DG = gradient_parameters.DG;

%% initialization

Nt = zeros(t,1); % list to store applicants number

% control parameters
phi = -10 + 20.*rand(ninfo,nPartcl);
phis = zeros(t,ninfo);
eps = -10 + 20.*rand(ninfo,nPartcl);
eps_arr = zeros(t,ninfo);
% F = zeros(1,2*ninfo);
% Rbar = 0; % Rbar

% perceptron parameters
P = ones([ninfo,1]);
w = 0;

% initiate average cumulative rewards
R_proposed_cum = zeros([t,1]);
R_prfct_cum = zeros(size(R_proposed_cum));
R_P_cum = zeros(size(R_proposed_cum));
R_all_cum = zeros(size(R_proposed_cum));

% initiante acceptance ratio
global_max_numA = zeros(t,1);
% global_max_R = 0;
numA_P = zeros(t,1);
numA_prfct = zeros(t,1);
ratioAs = zeros(t,1);
ratioAs_P = zeros(t,1);
ratioAs_prfct = zeros(t,1);
ratioAs_all = ones(t,1);

% initiate default probability
global_max_default_num = zeros(size(R_proposed_cum));
default_prob = zeros(size(R_proposed_cum));
default_num_P = zeros(size(R_proposed_cum));
default_prob_P = zeros(size(R_proposed_cum));
default_num_prfct = zeros(size(R_proposed_cum));
default_prob_prfct = zeros(size(R_proposed_cum));
default_num_all = zeros(size(R_proposed_cum));
default_prob_all = zeros(size(R_proposed_cum));

%% iterations

for t_idx = 1:t
%     workbar(t_idx/t)
    
    % generate applicants number & info
    N = randi([10000,20000],1);
%     N = 20000;
    Nt(t_idx) = N;
    nempty = ceil(nempty_prcnt*N*ninfo);

    % Personal information: N x ninfo
    [p,s] = random_apc_info(N, ninfo, nempty);
    
    % return probability
    return_varialbe = rand(N,1);
    
    %% choose all
    
    % calculate rewards
    R_all = zeros([N,1]);
    R_all(return_varialbe < p) = c+e;
    R_all(return_varialbe >= p) = -1+e;
    
    % calculate default probability
    default_num_all(t_idx) = sum(R_all == (-1+e));
    default_prob_all(t_idx) = sum(default_num_all)/sum(Nt);
	
    %% perfect decision
    
    % making decision
    A_prfct = (p >= 0.95);
    
    % calculate rewards
    R_prfct = zeros([N,1]);
    R_prfct(A_prfct==1 & return_varialbe < p) = c+e;
    R_prfct(A_prfct==1 & return_varialbe >= p) = -1+e;
    
    % calculate acceptance probability
    numA_prfct(t_idx) = sum(R_prfct>0); % number of acceptance
    ratioAs_prfct(t_idx) = sum(numA_prfct)/sum(Nt); % acceptance ratio
    
    % calculate default probability
    default_num_prfct(t_idx) = sum(R_prfct == (-1+e));
    default_prob_prfct(t_idx) = sum(default_num_prfct)/sum(numA_prfct);
    
    %% perceptron
    
    % calculate the decision value
    sP = s; sP(isnan(s)) = 0;
    OP = sum(sP.*P'+w,2);
    
    % making decision
    A_P = (OP > 0);
    
    % calculate rewards
    R_P = zeros([N,1]);
    R_P(A_P == 1 & return_varialbe < p) = c+e;
    R_P(A_P == 1 & return_varialbe >= p) = -1+e;
    
    % calculate acceptance probability
    numA_P(t_idx) = sum(A_P); % number of acceptance
    ratioAs_P(t_idx) = sum(numA_P)/sum(Nt); % acceptance ratio
    
    % calculate default probability
    default_num_P(t_idx) = sum(R_P == (-1+e));
    default_prob_P(t_idx) = sum(default_num_P)/sum(numA_P);
    
    % updating parameters
    neg_idx = find(A_P == 1 & return_varialbe >= p);
    pos_idx = find(A_P == 0 & return_varialbe < p);
    P = P - sum(sP(neg_idx,:))';
    w = w - numel(neg_idx);
    P = P + sum(sP(pos_idx,:))';
    w = w + numel(pos_idx);
    
    
    %% Proposed Approach
    
    % updating step size
    alpha = DG/sqrt(t_idx);
    
    if t_idx <= MSO_itr
        
        % gradient update for each particle
        R_sum = zeros(1,nPartcl);
        numA = zeros(1,nPartcl);
        default_num = zeros(1,nPartcl);
        parfor p_idx = 1:nPartcl

            % current particle
            phi_now = phi(:,p_idx);
            eps_now = eps(:,p_idx);
            z = [phi_now;eps_now];
            
            % calculate Q
            s_phi = s.*phi_now';
            Q = s_phi + eps_now'; Q(isnan(Q)) = 0;
                Qs = sign(Q); Q = abs(Q); Q(isinf(Q)) = realmax; Q = Qs.*Q;
                
            % policy pi
            exp_Q = exp(mean((k.*Q),2));
                exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;
                
            % pick the L form
            switch L_form
                case 'A'
                    pie = exp_Q./(1+exp_Q);
                case 'B'
                    pie = 1-(1./exp_Q);
                    pie(pie<0) = 0;
                case 'C'
                    pie = (2.*exp_Q./(1+exp_Q))-1;
            end

            % make decision
            decision_varialbe = rand(N,1);
            A = (decision_varialbe < pie);

            % calculate rewards
            R = zeros([N,1]);
            R(A == 1 & return_varialbe < p) = c+e;
            R(A == 1 & return_varialbe >= p) = -1+e;

            % index of the accepted applications
            Aid = find(A == 1);
            del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k);
                del_pi(del_pi == 0) = realmin;
                del_pi(del_pi == inf) = realmax;
                pie(pie == 0) = realmin;

            Rbar = sum(R_proposed_cum)/sum(Nt);

            deltaR = (R - Rbar);
            F = mean((del_pi./pie).*deltaR,1);
                Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;

            % update parameters
            z = z + alpha.*F';
                zs = sign(z); z = abs(z); z(isinf(z)) = realmax; z = zs.*z;
            phi(:,p_idx) = z(1:ninfo);
            eps(:,p_idx) = z(ninfo+1:end);

            % summing rewards
            R_sum(p_idx) = sum(R);

            % get acceptance and default numbers
            numA(p_idx) = sum(A);
            default_num(p_idx) = sum(R == (-1+e));

        end
    
        % Pick the maximum					 
        global_max_R = max(R_sum); % update global maximum
		global_max_idx = find(R_sum == global_max_R);
		global_max_phi = phi(:,global_max_idx(1));
        global_max_eps = eps(:,global_max_idx(1));
        
        % calculate acceptance probability
        global_max_numA(t_idx) = numA(global_max_idx(1));
        ratioAs(t_idx) = sum(global_max_numA)/sum(Nt);
        
        % calculate default probability
        global_max_default_num(t_idx) = default_num(global_max_idx(1));
        default_prob(t_idx) = ...
		    sum(global_max_default_num)/sum(global_max_numA);
        
        % update parameters
        [~,srt_idx] = sort(R_sum,'descend');
        phi(:,1:nkeep) = phi(:,srt_idx(1:nkeep));
        phi(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        eps(:,1:nkeep) = eps(:,srt_idx(1:nkeep));
        eps(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
		
    else
        
        pie = zeros(size(p));
        
        % continue the parameters
        phi_now = global_max_phi;
        eps_now = global_max_eps;
        z = [phi_now;eps_now];

        % calculate Q
        s_phi = s.*phi_now';
        Q = s_phi + eps_now'; Q(isnan(Q)) = 0;
            Qs = sign(Q); Q = abs(Q); Q(isinf(Q)) = realmax; Q = Qs.*Q;

        % policy pi
        exp_Q = exp(mean((k.*Q),2));
            exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

        % pick the L form
        switch L_form
            case 'A'
                pie = exp_Q./(1+exp_Q);
            case 'B'
                pie = 1-(1./exp_Q);
                pie(pie<0) = 0;
            case 'C'
                pie = (2.*exp_Q./(1+exp_Q))-1;
        end

        % make decision
        decision_varialbe = rand(N,1);
        A = (decision_varialbe < pie);

        % calculate rewards
        R = zeros([N,1]);
        R(A == 1 & return_varialbe < p) = c+e;
        R(A == 1 & return_varialbe >= p) = -1+e;

        % index of the accepted applications
        Aid = find(A == 1);
        del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k);
            del_pi(del_pi == 0) = realmin;
            del_pi(del_pi == inf) = realmax;
            pie(pie == 0) = realmin;

        Rbar = sum(R_proposed_cum)/sum(Nt);

        deltaR = (R - Rbar);
        F = mean((del_pi./pie).*deltaR,1);
            Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;

        % update parameters
        z = z + alpha.*F';
            zs = sign(z); z = abs(z); z(isinf(z)) = realmax; z = zs.*z;
        global_max_phi = z(1:ninfo);
        global_max_eps = z(ninfo+1:end);

        % summing rewards
        global_max_R = sum(R);

        % calculate acceptance probability
        global_max_numA(t_idx) = sum(A);
        ratioAs(t_idx) = sum(global_max_numA)/sum(Nt);
        
        % calculate default probability
        global_max_default_num(t_idx) = sum(R == (-1+e));
        default_prob(t_idx) = ...
            sum(global_max_default_num)/sum(global_max_numA);
        
    end
    
    % store maximum
    phis(t_idx,:) = global_max_phi';
    eps_arr(t_idx,:) = global_max_eps';
    
    %% cumulative rewards over time
    
    if t_idx == 1
        R_proposed_cum(t_idx) = global_max_R;
        R_prfct_cum(t_idx) = sum(R_prfct);
        R_P_cum(t_idx) = sum(R_P);
		R_all_cum(t_idx) = sum(R_all);
    else
        R_proposed_cum(t_idx) = ...
            (sum(global_max_R)+(R_proposed_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_prfct_cum(t_idx) = ...
            (sum(R_prfct)+(R_prfct_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_P_cum(t_idx) = ...
            (sum(R_P)+(R_P_cum(t_idx-1))*(t_idx-1))/t_idx;
		R_all_cum(t_idx) = ...
            (sum(R_all)+(R_all_cum(t_idx-1))*(t_idx-1))/t_idx;
    end

end

%% compile output

% compile cumulative rewards
R_cum.proposed = R_proposed_cum;
R_cum.perfect = R_prfct_cum;
R_cum.perceptron = R_P_cum;
R_cum.all = R_all_cum;

% compile acceptance ratio
acceptance.proposed = ratioAs;
acceptance.perfect = ratioAs_prfct;
acceptance.perceptron = ratioAs_P;
acceptance.all = ratioAs_all;

% compile default probability
default.proposed = default_prob;
default.perfect = default_prob_prfct;
default.perceptron = default_prob_P;
default.all = default_prob_all;

% compile parameters
parameters.proposed.phis = phis;
parameters.proposed.eps = eps_arr;
parameters.perceptron.P = P;
parameters.perceptron.w = w;

end

%% function to generate s

function [p,s] = random_apc_info(n_apcs, ninfo, nempty)

% generate s

% s1_prcnt = 0.2 + 0.2*rand(1);
% s1_num = ceil(s1_prcnt*n_apcs);
% s1 = 2.*rand(s1_num,ninfo);
% s2 = 2 + 2.*rand(n_apcs-s1_num,ninfo);
% s = [s1;s2];
% s = s(randperm(size(s, 1)), :);

n_bins = 100;
s1_prcnt = [linspace(0,1/n_bins,n_bins),0.5];%0.2 + 0.2*rand(1);
s1_num = floor(s1_prcnt*n_apcs);
s1_lb = linspace(0,4,n_bins+1);
s = 4.*rand(n_apcs,ninfo);
for i = 2:n_bins+1
    s(sum(s1_num(1:i-1))+1:sum(s1_num(1:i-1))+s1_num(i),:) = ...
        s1_lb(i-1) + (1./n_bins).*s(sum(s1_num(1:i-1))+1:sum(s1_num(1:i-1))+s1_num(i),:);
end
s = s(randperm(size(s, 1)), :);

%apcs_info = 4*ones([20000,1]);
% apcs_info = randi([0 4],n_apcs,ninfo);
% apcs_info = zeros(n_apcs,ninfo);
% s = 4 - abs(normrnd(0,1,n_apcs,ninfo));
% apcs_info = normrnd(3,1,n_apcs,ninfo);
%     apcs_info(apcs_info < 0) = 0;
%     apcs_info(apcs_info > 4) = 4;
% s = 4.*rand(n_apcs,ninfo);
% apcs_info(1:n_apcs,:) = 4;
%     disp(size(apcs_info));
%apcs_info = [apcs_info1 apcs_info2]

% calculate the probability
% p = (sum(s.*w,2) ./ sum((4.*w)));

p = mean(s,2)/4;
% p(p>0.6) = p(p>0.6)+0.225;
% p(p>1) = 1;
% p(p<0.95 & p>0.5) = 1;

% p = 1./(1+exp(-3.*(mean(s,2)-2)));

% p(p>0.475) = p(p>0.475)+0.5;
%     p(p>1) = 1;
% p(p<=0.475) = p(p<=0.475)-0.2;
%     p(p<0) = 0;
% p = (2.*exp(mean(s,2))./(1+exp(mean(s,2))))-1;
%     p = sum(s,2);

% h = s(:,1)+4.*s(:,2);
% p = zeros(size(h));
% p(h>10) = 1;
% p = h./20;
% p(h>10) = 0.95;
% p = s>=2;

emty_idx = randperm(numel(s(:)),nempty);
s(emty_idx) = NaN;

end

%% function to calcualte the derevative of the probability function

function del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k)

del_pi = zeros([N, 2*ninfo]);

exp_Q = exp(k.*Q); % Q: Nxninfo
    exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

exp_plus = exp_Q + 1;
exp_sqr = exp_plus.^2;
    exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;

switch L_form
    case 'A'
        par_del = exp_Q./exp_sqr;
    case 'B'
        par_del = k./exp_Q;
        par_del(par_del>k) = k;
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






