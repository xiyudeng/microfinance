function [R_cum,acceptance,default,parameters] = MFI_simulation...
    (case_option,MS_parameters,gradient_parameters,train_lim,...
    data,rndm)

if ~rndm
    rng(1)
end
%% case option

% number of applicants each period
Nlim = case_option.Nlim; % lower and upper limit

% parameters to generate s and p
% s_a = case_option.s_a;
% p_func = case_option.p_func;

% load s and p data
% load(['artificial_data_a_',num2str(case_option.s_a),...
%     '_pFunc_',case_option.p_func.type,num2str(case_option.p_func.const),...
%     '.mat'],'s','p')
% s_data = s;
% p_data = p;

% s, p, and A data
s_data = data.s;
p_data = data.p;
status_data = data.status;

% reward rule
c = case_option.c; % interest rate
e = case_option.e; % encouragement

% information parameters
ninfo = case_option.ninfo; % number of information for each applicant
nempty_prcnt = case_option.nempty; % percentage of empty entries

% number of period
t = case_option.t;

%% MS parameters

nPartcl = MS_parameters.nPartcl;            % population size
nkeep = MS_parameters.nkeep;              % number to keep
MSO_itr = MS_parameters.itr;           % multi start iterations

%% gradient optimization parameters

% form of pi
% L_form = gradient_parameters.L_form;
k = gradient_parameters.k;

% step size constant
% DG = gradient_parameters.DG;
DG_A = gradient_parameters.DG.A;
DG_B = gradient_parameters.DG.B;
DG_C = gradient_parameters.DG.C;

%% initialization

Nt = zeros(t,1); % list to store applicants number

% control parameters
% phi = -10 + 20.*rand(ninfo,nPartcl);
% phis = zeros(t,ninfo);
% eps = -10 + 20.*rand(ninfo,nPartcl);
% eps_arr = zeros(t,ninfo);
phi_A = -10 + 20.*rand(ninfo,nPartcl);
phis_A = zeros(t,ninfo);
eps_A = -10 + 20.*rand(ninfo,nPartcl);
eps_arr_A = zeros(t,ninfo);
phi_B = -10 + 20.*rand(ninfo,nPartcl);
phis_B = zeros(t,ninfo);
eps_B = -10 + 20.*rand(ninfo,nPartcl);
eps_arr_B = zeros(t,ninfo);
phi_C = -10 + 20.*rand(ninfo,nPartcl);
phis_C = zeros(t,ninfo);
eps_C = -10 + 20.*rand(ninfo,nPartcl);
eps_arr_C = zeros(t,ninfo);

% F = zeros(1,2*ninfo);
% Rbar = 0; % Rbar

% perceptron parameters
P = ones([ninfo,1]);
w = 0;

% perfect decision parameter
dec_lim = zeros(t,1); % initial threshold
dec_lim(2) = 0.5; % second update of the threshold

% neural network net
loan_mdl_N = patternnet(1);
loan_mdl_N.trainParam.showWindow = 0;

% initiate average cumulative rewards
R_proposed_cum_A = zeros([t,1]);
R_proposed_cum_B = zeros(size(R_proposed_cum_A));
R_proposed_cum_C = zeros(size(R_proposed_cum_A));
R_prfct1_cum = zeros(size(R_proposed_cum_A));
R_prfct2_cum = zeros(size(R_proposed_cum_A));
R_pred_cum = zeros(size(R_proposed_cum_A));
R_P_cum = zeros(size(R_proposed_cum_A));
R_T_cum = zeros(size(R_proposed_cum_A));
R_svm_cum = zeros(size(R_proposed_cum_A));
R_L_cum = zeros(size(R_proposed_cum_A));
R_N_cum = zeros(size(R_proposed_cum_A));
R_all_cum = zeros(size(R_proposed_cum_A));

% initiante acceptance ratio
global_max_numA_A = zeros(t,1);
global_max_numA_B = zeros(t,1);
global_max_numA_C = zeros(t,1);
% global_max_R = 0;
numA_P = zeros(t,1);
numA_T = zeros(t,1);
numA_svm = zeros(t,1);
numA_L = zeros(t,1);
numA_N = zeros(t,1);
numA_prfct1 = zeros(t,1);
numA_prfct2 = zeros(t,1);
numA_pred = zeros(t,1);
ratioAs_A = zeros(t,1);
ratioAs_B = zeros(t,1);
ratioAs_C = zeros(t,1);
ratioAs_P = zeros(t,1);
ratioAs_T = zeros(t,1);
ratioAs_svm = zeros(t,1);
ratioAs_L = zeros(t,1);
ratioAs_N = zeros(t,1);
ratioAs_prfct1 = zeros(t,1);
ratioAs_prfct2 = zeros(t,1);
ratioAs_pred = zeros(t,1);
ratioAs_all = ones(t,1);

% initiate default probability
global_max_default_num_A = zeros(size(R_proposed_cum_A));
default_prob_A = zeros(size(R_proposed_cum_A));
global_max_default_num_B = zeros(size(R_proposed_cum_A));
default_prob_B = zeros(size(R_proposed_cum_A));
global_max_default_num_C = zeros(size(R_proposed_cum_A));
default_prob_C = zeros(size(R_proposed_cum_A));
default_num_P = zeros(size(R_proposed_cum_A));
default_prob_P = zeros(size(R_proposed_cum_A));
default_num_T = zeros(size(R_proposed_cum_A));
default_prob_T = zeros(size(R_proposed_cum_A));
default_num_svm = zeros(size(R_proposed_cum_A));
default_prob_svm = zeros(size(R_proposed_cum_A));
default_num_L = zeros(size(R_proposed_cum_A));
default_prob_L = zeros(size(R_proposed_cum_A));
default_num_N = zeros(size(R_proposed_cum_A));
default_prob_N = zeros(size(R_proposed_cum_A));
default_num_prfct1 = zeros(size(R_proposed_cum_A));
default_prob_prfct1 = zeros(size(R_proposed_cum_A));
default_num_prfct2 = zeros(size(R_proposed_cum_A));
default_prob_prfct2 = zeros(size(R_proposed_cum_A));
default_num_pred = zeros(size(R_proposed_cum_A));
default_prob_pred = zeros(size(R_proposed_cum_A));
default_num_all = zeros(size(R_proposed_cum_A));
default_prob_all = zeros(size(R_proposed_cum_A));

%% iterations

for t_idx = 1:t
%     workbar(t_idx/t)
    
    % generate applicants number & info
    N = randi(Nlim,1);
%     N = 20000;
    Nt(t_idx) = N; sum_Nt = sum(Nt);
    nempty = ceil(nempty_prcnt*N*ninfo);

    % Personal information: N x ninfo
%     [p,s] = random_apc_info(N, ninfo, nempty, s_a, p_func);
    % picking data
    s_id = randi(numel(p_data),[N,1]);
    s = s_data(s_id,1:ninfo);
    p = p_data(s_id);
    LoanStatus = status_data(s_id);
    
    % assign empty
    emty_idx = randperm(numel(s(:)),nempty);
    s(emty_idx) = NaN;
    
    % return probability
%     return_varialbe = rand(N,1);
    
    %% choose all
    
    % calculate rewards
    R_all = zeros([N,1]);
    R_all(LoanStatus) = c+e;
    R_all(~LoanStatus) = -1+e;
    
    % calculate default probability
    default_num_all(t_idx) = sum(R_all == (-1+e));
    default_prob_all(t_idx) = sum(default_num_all)/sum_Nt;
	
    %% perfect decision
    
    % set threshold
    if t_idx >= 3
        % update the threshold
        dec_lim(t_idx) = dec_lim(t_idx-1) + ...
            ((R_prfct1_mean - R_prfct1_mean_prev)/(c+e));
        % store previous cumulative rewards
        R_prfct1_mean_prev = R_prfct1_mean; 
    elseif t_idx == 2
        % store previous cumulative rewards
        R_prfct1_mean_prev = R_prfct1_mean;
    end
    
    % making decision
    A_prfct1 = (p >= dec_lim(t_idx));
    
    % calculate rewards
    R_prfct1 = zeros([N,1]);
    R_prfct1(A_prfct1 & LoanStatus) = c+e;
    R_prfct1(A_prfct1 & ~LoanStatus) = -1+e;
    R_prfct1_mean = mean(R_prfct1);
    
    % calculate acceptance probability
    numA_prfct1(t_idx) = sum(R_prfct1>0); % number of acceptance
    ratioAs_prfct1(t_idx) = sum(numA_prfct1)/sum_Nt; % acceptance ratio
    
    % calculate default probability
    default_num_prfct1(t_idx) = sum(R_prfct1 == (-1+e));
    default_prob_prfct1(t_idx) = sum(default_num_prfct1)/sum(numA_prfct1);
    
    %% perfect decision constant limit
    
    % making decision
%     A_prfct2 = ((((c+1).*p)+e-1)>=0);
    A_prfct2 = (p >= ((1-e)/(1+c)));
    
    % calculate rewards
    R_prfct2 = zeros([N,1]);
    R_prfct2(A_prfct2 & LoanStatus) = c+e;
    R_prfct2(A_prfct2 & ~LoanStatus) = -1+e;
    
    % calculate acceptance probability
    numA_prfct2(t_idx) = sum(R_prfct2>0); % number of acceptance
    ratioAs_prfct2(t_idx) = sum(numA_prfct2)/sum_Nt; % acceptance ratio
    
    % calculate default probability
    default_num_prfct2(t_idx) = sum(R_prfct2 == (-1+e));
    default_prob_prfct2(t_idx) = sum(default_num_prfct2)/sum(numA_prfct2);
    
    %% Prediction Approach
    
    if t_idx == 1
        
        % calculate rewards
        R_pred = zeros([N,1]);
        R_pred(LoanStatus) = c+e;
        R_pred(~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_pred(t_idx) = N; % number of acceptance
        ratioAs_pred(t_idx) = 1; % acceptance ratio

        % calculate default probability
        default_num_pred(t_idx) = sum(R_pred == (-1+e));
        default_prob_pred(t_idx) = sum(default_num_pred)/sum(numA_pred);
        
        % store data for interpolation
        s_interp_pred = s(1:Nlim(1),:); 
            s_interp_pred(isnan(s_interp_pred)) = 0;
        p_interp_pred = p(1:Nlim(1));
        
    else
        
        if t_idx <= train_lim
        
            % interpolation model
            loan_mdl_pred = fit(mean(s_interp_pred,2),p_interp_pred,...
                'gauss1');

            % store data for next interpolation
            data_replc_idx = randperm(Nlim(1),Nlim(1)/10);
            s_interp_pred(data_replc_idx,:) = s(data_replc_idx,:);
                s_interp_pred(isnan(s_interp_pred)) = 0;
            p_interp_pred(data_replc_idx) = p(data_replc_idx);
        
        end
        
        % making decision
        p_pred = loan_mdl_pred(mean(s,2));
        A_pred = ((((c+1).*p_pred)+e-1)>=0);

        % calculate rewards
        R_pred = zeros([N,1]);
        R_pred(A_pred & LoanStatus) = c+e;
        R_pred(A_pred & ~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_pred(t_idx) = sum(A_pred); % number of acceptance
        ratioAs_pred(t_idx) = sum(numA_pred)/sum_Nt; % acceptance ratio

        % calculate default probability
        default_num_pred(t_idx) = sum(R_pred == (-1+e));
        default_prob_pred(t_idx) = sum(default_num_pred)/sum(numA_pred);
        
    end
    
    %% perceptron
    
    % calculate the decision value
    sP = s; sP(isnan(s)) = 0;
    OP = sum(sP.*P'+w,2);
    
    % making decision
    A_P = (OP > 0);
    
    % calculate rewards
    R_P = zeros([N,1]);
    R_P(A_P & LoanStatus) = c+e;
    R_P(A_P & ~LoanStatus) = -1+e;
    
    % calculate acceptance probability
    numA_P(t_idx) = sum(A_P); % number of acceptance
    ratioAs_P(t_idx) = sum(numA_P)/sum_Nt; % acceptance ratio
    
    % calculate default probability
    default_num_P(t_idx) = sum(R_P == (-1+e));
    default_prob_P(t_idx) = sum(default_num_P)/sum(numA_P);
    
    % updating parameters
    neg_idx = find(A_P & ~LoanStatus);
    pos_idx = find(~A_P & LoanStatus);
    P = P - sum(sP(neg_idx,:))';
    w = w - numel(neg_idx);
    P = P + sum(sP(pos_idx,:))';
    w = w + numel(pos_idx);
    
    
    
    %% Decision Tree
    
    if t_idx == 1
        
        % calculate rewards
        R_T = zeros([N,1]);
        R_T(LoanStatus) = c+e;
        R_T(~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_T(t_idx) = N; % number of acceptance
        ratioAs_T(t_idx) = 1; % acceptance ratio

        % calculate default probability
        default_num_T(t_idx) = sum(R_T == (-1+e));
        default_prob_T(t_idx) = sum(default_num_T)/sum(numA_T);
        
        % store data for training
        s_train_T = s(1:Nlim(1),:);
            s_train_T(isnan(s_train_T)) = 0;
        dec_train_T = LoanStatus(1:Nlim(1));
        
    else
        
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_T = fitctree(s_train_T,dec_train_T,...
                'OptimizeHyperparameters','auto',...
                'HyperparameterOptimizationOptions',...
                struct('ShowPlots',false,'Verbose',0,'UseParallel',true));

            % store data for training
            data_replc_idx = randperm(Nlim(1),Nlim(1)/10);
            s_train_T(data_replc_idx,:) = s(data_replc_idx,:);
                s_train_T(isnan(s_train_T)) = 0;
            dec_train_T(data_replc_idx) = LoanStatus(data_replc_idx);
        
        end
        
        % making decision
        A_T = predict(loan_mdl_T,s);

        % calculate rewards
        R_T = zeros([N,1]);
        R_T(A_T & LoanStatus) = c+e;
        R_T(A_T & ~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_T(t_idx) = sum(A_T); % number of acceptance
        ratioAs_T(t_idx) = sum(numA_T)/sum_Nt; % acceptance ratio

        % calculate default probability
        default_num_T(t_idx) = sum(R_T == (-1+e));
        default_prob_T(t_idx) = sum(default_num_T)/sum(numA_T);
        
    end
    
    %% Support Vector Machine (SVM)
    
    if t_idx == 1
        
        % calculate rewards
        R_svm = zeros([N,1]);
        R_svm(LoanStatus) = c+e;
        R_svm(~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_svm(t_idx) = N; % number of acceptance
        ratioAs_svm(t_idx) = 1; % acceptance ratio

        % calculate default probability
        default_num_svm(t_idx) = sum(R_svm == (-1+e));
        default_prob_svm(t_idx) = sum(default_num_svm)/sum(numA_svm);
        
        % store data for training
        s_train_svm = s(1:Nlim(1),:);
            s_train_svm(isnan(s_train_svm)) = 0;
        dec_train_svm = LoanStatus(1:Nlim(1));
        
    else
        
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_svm = fitclinear(s_train_svm,dec_train_svm,...
                'OptimizeHyperparameters','auto','Learner','svm',...
                'HyperparameterOptimizationOptions',...
                struct('ShowPlots',false,'Verbose',0,'UseParallel',true));

            % store data for training
            data_replc_idx = randperm(Nlim(1),Nlim(1)/10);
            s_train_svm(data_replc_idx,:) = s(data_replc_idx,:);
                s_train_svm(isnan(s_train_svm)) = 0;
            dec_train_svm(data_replc_idx) = LoanStatus(data_replc_idx);
        
        end
        
        % making decision
        A_svm = predict(loan_mdl_svm,s);

        % calculate rewards
        R_svm = zeros([N,1]);
        R_svm(A_svm & LoanStatus) = c+e;
        R_svm(A_svm & ~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_svm(t_idx) = sum(A_svm); % number of acceptance
        ratioAs_svm(t_idx) = sum(numA_svm)/sum_Nt; % acceptance ratio

        % calculate default probability
        default_num_svm(t_idx) = sum(R_svm == (-1+e));
        default_prob_svm(t_idx) = sum(default_num_svm)/sum(numA_svm);
        
    end
    
    
    %% Logistic Regression
    
    if t_idx == 1
        
        % calculate rewards
        R_L = zeros([N,1]);
        R_L(LoanStatus) = c+e;
        R_L(~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_L(t_idx) = N; % number of acceptance
        ratioAs_L(t_idx) = 1; % acceptance ratio

        % calculate default probability
        default_num_L(t_idx) = sum(R_L == (-1+e));
        default_prob_L(t_idx) = sum(default_num_L)/sum(numA_L);
        
        % store data for training
        s_train_L = s(1:Nlim(1),:);
            s_train_L(isnan(s_train_L)) = 0;
        dec_train_L = LoanStatus(1:Nlim(1));
        
    else
        
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_L = fitclinear(s_train_L,dec_train_L,...
                'OptimizeHyperparameters','auto','Learner','logistic',...
                'HyperparameterOptimizationOptions',...
                struct('ShowPlots',false,'Verbose',0,'UseParallel',true));

            % store data for training
            data_replc_idx = randperm(Nlim(1),Nlim(1)/10);
            s_train_L(data_replc_idx,:) = s(data_replc_idx,:);
                s_train_L(isnan(s_train_L)) = 0;
            dec_train_L(data_replc_idx) = LoanStatus(data_replc_idx);
        
        end
        
        % making decision
        A_L = predict(loan_mdl_L,s);

        % calculate rewards
        R_L = zeros([N,1]);
        R_L(A_L & LoanStatus) = c+e;
        R_L(A_L & ~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_L(t_idx) = sum(A_L); % number of acceptance
        ratioAs_L(t_idx) = sum(numA_L)/sum_Nt; % acceptance ratio

        % calculate default probability
        default_num_L(t_idx) = sum(R_L == (-1+e));
        default_prob_L(t_idx) = sum(default_num_L)/sum(numA_L);
        
    end
    
    
    %% Neural Network
    
    if t_idx == 1
        
        % calculate rewards
        R_N = zeros([N,1]);
        R_N(LoanStatus) = c+e;
        R_N(~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_N(t_idx) = N; % number of acceptance
        ratioAs_N(t_idx) = 1; % acceptance ratio

        % calculate default probability
        default_num_N(t_idx) = sum(R_T == (-1+e));
        default_prob_N(t_idx) = sum(default_num_T)/sum(numA_T);
        
        % store data for training
        s_train_N = s;
            s_train_N(isnan(s_train_N)) = 0;
        dec_train_N = LoanStatus;
        
    else
        
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_N = train(loan_mdl_N,s_train_N',dec_train_N');

            % store data for training
            s_train_N = s;
                s_train_N(isnan(s_train_N)) = 0;
            dec_train_N = LoanStatus;
        
        end
        
        % making decision
        A_N = (loan_mdl_N(s')>0.5)';

        % calculate rewards
        R_N = zeros([N,1]);
        R_N(A_N & LoanStatus) = c+e;
        R_N(A_N & ~LoanStatus) = -1+e;
        
        % calculate acceptance probability
        numA_N(t_idx) = sum(A_N); % number of acceptance
        ratioAs_N(t_idx) = sum(numA_N)/sum_Nt; % acceptance ratio

        % calculate default probability
        default_num_N(t_idx) = sum(R_N == (-1+e));
        default_prob_N(t_idx) = sum(default_num_N)/sum(numA_T);
        
    end
    
    
    %% Proposed Approach
    
    % updating step size
    alpha_A = DG_A/sqrt(t_idx);
    alpha_B = DG_B/sqrt(t_idx);
    alpha_C = DG_C/sqrt(t_idx);
    
    if t_idx <= MSO_itr
        
        % gradient update for each particle
        R_sum_A = zeros(1,nPartcl);
        numA_A = zeros(1,nPartcl);
        default_num_A = zeros(1,nPartcl);
        R_sum_B = zeros(1,nPartcl);
        numA_B = zeros(1,nPartcl);
        default_num_B = zeros(1,nPartcl);
        R_sum_C = zeros(1,nPartcl);
        numA_C = zeros(1,nPartcl);
        default_num_C = zeros(1,nPartcl);
        parfor p_idx = 1:nPartcl
            
            % Case A
            
            % current particle
            phi_now = phi_A(:,p_idx);
            eps_now = eps_A(:,p_idx);
            
            % perform gradient ascent
            [phi_A(:,p_idx),eps_A(:,p_idx),R_sum_A(p_idx),...
                numA_A(p_idx),default_num_A(p_idx)] = ...
                gradient_ascent(N,s,ninfo,LoanStatus,k,'A',...
                c,e,phi_now,eps_now,alpha_A,R_proposed_cum_A,sum_Nt);
            
            % Case B
            
            % current particle
            phi_now = phi_B(:,p_idx);
            eps_now = eps_B(:,p_idx);
            
            % perform gradient ascent
            [phi_B(:,p_idx),eps_B(:,p_idx),R_sum_B(p_idx),...
                numA_B(p_idx),default_num_B(p_idx)] = ...
                gradient_ascent(N,s,ninfo,LoanStatus,k,'B',...
                c,e,phi_now,eps_now,alpha_B,R_proposed_cum_B,sum_Nt);
            
            % Case C
            
            % current particle
            phi_now = phi_C(:,p_idx);
            eps_now = eps_C(:,p_idx);
            
            % perform gradient ascent
            [phi_C(:,p_idx),eps_C(:,p_idx),R_sum_C(p_idx),...
                numA_C(p_idx),default_num_C(p_idx)] = ...
                gradient_ascent(N,s,ninfo,LoanStatus,k,'C',...
                c,e,phi_now,eps_now,alpha_C,R_proposed_cum_C,sum_Nt);
            
        end
    
        % Pick the maximum					 
        global_max_R_A = max(R_sum_A); % update global maximum
		global_max_idx_A = find(R_sum_A == global_max_R_A);
		global_max_phi_A = phi_A(:,global_max_idx_A(1));
        global_max_eps_A = eps_A(:,global_max_idx_A(1));
        global_max_R_B = max(R_sum_B); % update global maximum
		global_max_idx_B = find(R_sum_B == global_max_R_B);
		global_max_phi_B = phi_B(:,global_max_idx_B(1));
        global_max_eps_B = eps_B(:,global_max_idx_B(1));
        global_max_R_C = max(R_sum_C); % update global maximum
		global_max_idx_C = find(R_sum_C == global_max_R_C);
		global_max_phi_C = phi_C(:,global_max_idx_C(1));
        global_max_eps_C = eps_C(:,global_max_idx_C(1));
        
        % calculate acceptance probability
        global_max_numA_A(t_idx) = numA_A(global_max_idx_A(1));
        ratioAs_A(t_idx) = sum(global_max_numA_A)/sum_Nt;
        global_max_numA_B(t_idx) = numA_B(global_max_idx_B(1));
        ratioAs_B(t_idx) = sum(global_max_numA_B)/sum_Nt;
        global_max_numA_C(t_idx) = numA_C(global_max_idx_C(1));
        ratioAs_C(t_idx) = sum(global_max_numA_C)/sum_Nt;
        
        % calculate default probability
        global_max_default_num_A(t_idx) = ...
            default_num_A(global_max_idx_A(1));
        default_prob_A(t_idx) = ...
		    sum(global_max_default_num_A)/sum(global_max_numA_A);
        global_max_default_num_B(t_idx) = ...
            default_num_B(global_max_idx_B(1));
        default_prob_B(t_idx) = ...
		    sum(global_max_default_num_B)/sum(global_max_numA_B);
        global_max_default_num_C(t_idx) = ...
            default_num_C(global_max_idx_C(1));
        default_prob_C(t_idx) = ...
		    sum(global_max_default_num_C)/sum(global_max_numA_C);
        
        % update parameters
        [~,srt_idx] = sort(R_sum_A,'descend');
        phi_A(:,1:nkeep) = phi_A(:,srt_idx(1:nkeep));
        phi_A(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        eps_A(:,1:nkeep) = eps_A(:,srt_idx(1:nkeep));
        eps_A(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        [~,srt_idx] = sort(R_sum_B,'descend');
        phi_B(:,1:nkeep) = phi_B(:,srt_idx(1:nkeep));
        phi_B(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        eps_B(:,1:nkeep) = eps_B(:,srt_idx(1:nkeep));
        eps_B(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        [~,srt_idx] = sort(R_sum_C,'descend');
        phi_C(:,1:nkeep) = phi_C(:,srt_idx(1:nkeep));
        phi_C(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
        eps_C(:,1:nkeep) = eps_C(:,srt_idx(1:nkeep));
        eps_C(:,nkeep+1:end) = -10 + 20.*rand(ninfo,nPartcl-nkeep);
		
    else
        
        % case A
        
        % continue the parameters
        phi_now = global_max_phi_A;
        eps_now = global_max_eps_A;
        
        % perform gradient ascent
        [global_max_phi_A,global_max_eps_A,global_max_R_A,...
            global_max_numA_A(t_idx),global_max_default_num_A(t_idx)] = ...
            gradient_ascent(N,s,ninfo,LoanStatus,k,'A',...
            c,e,phi_now,eps_now,alpha_A,R_proposed_cum_A,sum_Nt);
        
        % calculate acceptance probability
        ratioAs_A(t_idx) = sum(global_max_numA_A)/sum_Nt;
        
        % calculate default probability
        default_prob_A(t_idx) = ...
            sum(global_max_default_num_A)/sum(global_max_numA_A);
        
        % case B
        
        % continue the parameters
        phi_now = global_max_phi_B;
        eps_now = global_max_eps_B;
        
        % perform gradient ascent
        [global_max_phi_B,global_max_eps_B,global_max_R_B,...
            global_max_numA_B(t_idx),global_max_default_num_B(t_idx)] = ...
            gradient_ascent(N,s,ninfo,LoanStatus,k,'B',...
            c,e,phi_now,eps_now,alpha_B,R_proposed_cum_B,sum_Nt);
        
        % calculate acceptance probability
        ratioAs_B(t_idx) = sum(global_max_numA_B)/sum_Nt;
        
        % calculate default probability
        default_prob_B(t_idx) = ...
            sum(global_max_default_num_B)/sum(global_max_numA_B);
        
        % case C
        
        % continue the parameters
        phi_now = global_max_phi_C;
        eps_now = global_max_eps_C;
        
        % perform gradient ascent
        [global_max_phi_C,global_max_eps_C,global_max_R_C,...
            global_max_numA_C(t_idx),global_max_default_num_C(t_idx)] = ...
            gradient_ascent(N,s,ninfo,LoanStatus,k,'C',...
            c,e,phi_now,eps_now,alpha_C,R_proposed_cum_C,sum_Nt);
        
        % calculate acceptance probability
        ratioAs_C(t_idx) = sum(global_max_numA_C)/sum_Nt;
        
        % calculate default probability
        default_prob_C(t_idx) = ...
            sum(global_max_default_num_C)/sum(global_max_numA_C);
        
    end
    
    % store maximum
    phis_A(t_idx,:) = global_max_phi_A';
    eps_arr_A(t_idx,:) = global_max_eps_A';
    phis_B(t_idx,:) = global_max_phi_B';
    eps_arr_B(t_idx,:) = global_max_eps_B';
    phis_C(t_idx,:) = global_max_phi_C';
    eps_arr_C(t_idx,:) = global_max_eps_C';
    
    %% cumulative rewards over time
    
    if t_idx == 1
        R_proposed_cum_A(t_idx) = global_max_R_A;
        R_proposed_cum_B(t_idx) = global_max_R_B;
        R_proposed_cum_C(t_idx) = global_max_R_C;
        R_prfct1_cum(t_idx) = sum(R_prfct1);
        R_prfct2_cum(t_idx) = sum(R_prfct2);
        R_pred_cum(t_idx) = sum(R_pred);
        R_P_cum(t_idx) = sum(R_P);
        R_T_cum(t_idx) = sum(R_T);
        R_svm_cum(t_idx) = sum(R_svm);
        R_L_cum(t_idx) = sum(R_L);
        R_N_cum(t_idx) = sum(R_N);
		R_all_cum(t_idx) = sum(R_all);
    else
        R_proposed_cum_A(t_idx) = ...
            (sum(global_max_R_A)+(R_proposed_cum_A(t_idx-1))*(t_idx-1))/...
            t_idx;
        R_proposed_cum_B(t_idx) = ...
            (sum(global_max_R_B)+(R_proposed_cum_B(t_idx-1))*(t_idx-1))/...
            t_idx;
        R_proposed_cum_C(t_idx) = ...
            (sum(global_max_R_C)+(R_proposed_cum_C(t_idx-1))*(t_idx-1))/...
            t_idx;
        R_prfct1_cum(t_idx) = ...
            (sum(R_prfct1)+(R_prfct1_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_prfct2_cum(t_idx) = ...
            (sum(R_prfct2)+(R_prfct2_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_pred_cum(t_idx) = ...
            (sum(R_pred)+(R_pred_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_P_cum(t_idx) = ...
            (sum(R_P)+(R_P_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_T_cum(t_idx) = ...
            (sum(R_T)+(R_T_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_svm_cum(t_idx) = ...
            (sum(R_svm)+(R_svm_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_L_cum(t_idx) = ...
            (sum(R_L)+(R_L_cum(t_idx-1))*(t_idx-1))/t_idx;
        R_N_cum(t_idx) = ...
            (sum(R_N)+(R_N_cum(t_idx-1))*(t_idx-1))/t_idx;
		R_all_cum(t_idx) = ...
            (sum(R_all)+(R_all_cum(t_idx-1))*(t_idx-1))/t_idx;
    end

end

%% compile output

% compile cumulative rewards
R_cum.proposed.A = R_proposed_cum_A;
R_cum.proposed.B = R_proposed_cum_B;
R_cum.proposed.C = R_proposed_cum_C;
R_cum.perfect.updating = R_prfct1_cum;
R_cum.perfect.constant = R_prfct2_cum;
R_cum.p_prediction = R_pred_cum;
R_cum.perceptron = R_P_cum;
R_cum.FitTree = R_T_cum;
R_cum.SVM = R_svm_cum;
R_cum.LogisticRegression = R_L_cum;
R_cum.neural = R_N_cum;
R_cum.all = R_all_cum;

% compile acceptance ratio
acceptance.proposed.A = ratioAs_A;
acceptance.proposed.B = ratioAs_B;
acceptance.proposed.C = ratioAs_C;
acceptance.perfect.updating = ratioAs_prfct1;
acceptance.perfect.constant = ratioAs_prfct2;
acceptance.p_prediction = ratioAs_pred;
acceptance.perceptron = ratioAs_P;
acceptance.FitTree = ratioAs_T;
acceptance.SVM = ratioAs_svm;
acceptance.LogisticRegression = ratioAs_L;
acceptance.neural = ratioAs_N;
acceptance.all = ratioAs_all;

% compile default probability
default.proposed.A = default_prob_A;
default.proposed.B = default_prob_B;
default.proposed.C = default_prob_C;
default.perfect.updating = default_prob_prfct1;
default.perfect.constant = default_prob_prfct2;
default.p_prediction = default_prob_pred;
default.perceptron = default_prob_P;
default.FitTree = default_prob_T;
default.SVM = default_prob_svm;
default.LogisticRegression = default_prob_L;
default.neural = default_prob_N;
default.all = default_prob_all;

% compile parameters
parameters.proposed.A.phis = phis_A;
parameters.proposed.A.eps = eps_arr_A;
parameters.proposed.B.phis = phis_B;
parameters.proposed.B.eps = eps_arr_B;
parameters.proposed.C.phis = phis_C;
parameters.proposed.C.eps = eps_arr_C;
parameters.perfect.updating.threshold = dec_lim;
parameters.perceptron.p_prediction = loan_mdl_pred;
parameters.perceptron.P = P;
parameters.perceptron.w = w;
parameters.FitTree = loan_mdl_T;
parameters.SVM = loan_mdl_svm;
parameters.LogisticRegression = loan_mdl_L;
parameters.neural = loan_mdl_N;

end

%% function to generate s artificialy on the go

% function [p,s] = random_apc_info(n_apcs, ninfo, nempty, s_a, p_func)
% 
% % generate s
% n_bins = min(ceil(n_apcs/5),100);
% s1_prcnt = linspace(s_a,(2/n_bins)-s_a,n_bins);
% s1_num = floor(s1_prcnt*n_apcs);
% s1_lb = linspace(0,4,n_bins);
% s = 4.*rand(n_apcs,ninfo);
% for i = 2:n_bins
%     s(sum(s1_num(1:i-1))+1:sum(s1_num(1:i-1))+s1_num(i),:) = ...
%         s1_lb(i-1) + (1./n_bins).*s(sum(s1_num(1:i-1))+...
%         1:sum(s1_num(1:i-1))+s1_num(i),:);
% end
% s = s(randperm(size(s, 1)), :);
% 
% % calculate the probability
% switch p_func.type
%     case 'linear'
%         p = mean(s,2)./4;
%     case 'quadratic'
%         p = (-1/16).*mean(s,2).^2 + (1/2).*mean(s,2);
%     case 'exponential'
%         q = p_func.const.*mean(s,2)-4;
%         p = (exp(q)) ./ (1+exp(q));
% end
% 
% % assign empty
% emty_idx = randperm(numel(s(:)),nempty);
% s(emty_idx) = NaN;
% 
% end

%% function for the gradient ascent step

function [phi_now,eps_now,R_sum,numA,default_num] = ...
    gradient_ascent(N,s,ninfo,LoanStatus,k,L_form,c,e,...
    phi_now,eps_now,alpha,R_proposed_cum,sum_Nt)

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
%                     pie(pie<0) = 0;
    case 'C'
        pie = (2.*exp_Q./(1+exp_Q))-1;
end
pie(pie == 0) = realmin;

% make decision
decision_varialbe = rand(N,1);
A = (decision_varialbe < pie);

% calculate rewards
R = zeros([N,1]);
R(A & LoanStatus) = c+e;
R(A & ~LoanStatus) = -1+e;

% index of the accepted applications
Aid = find(A == 1);
del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k);

Rbar = sum(R_proposed_cum)/sum_Nt;

deltaR = (R - Rbar);
F = mean((del_pi./pie).*deltaR,1);
    Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;

% update parameters
z = [phi_now;eps_now];
z = z + alpha.*F';
    zs = sign(z); z = abs(z); z(isinf(z)) = realmax; z = zs.*z;
phi_now = z(1:ninfo);
eps_now = z(ninfo+1:end);

% summing rewards
R_sum = sum(R);

% get acceptance and default numbers
numA = sum(A);
default_num = sum(R == (-1+e));

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
    del_pi(del_pi==0) = realmin; del_pi(isinf(del_pi)) = realmax;
    del_pi(isnan(del_pi)) = realmin;
    
end







