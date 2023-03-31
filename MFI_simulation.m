function [R_cum,acceptance,default,parameters,Nt,comp_time] = MFI_simulation...
    (case_option,MS_parameters,gradient_parameters,train_lim,...
    data,rndm,isgroup)

if ~rndm
    rng(1)
end
%% case option

% number of applicants each period
Nlim = case_option.Nlim; % lower and upper limit

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

% L_form parameter
k = gradient_parameters.k;

% step size constant
DG_C = gradient_parameters.DG.C;

%% initialization

Nt = zeros(t,1); % list to store applicants number

% control parameters
z_lb = -10; z_rng = 20;
phi_C = zeros(ninfo,nPartcl);
phis_C = zeros(t,ninfo);
eps_C = ones(ninfo,nPartcl);
eps_arr_C = zeros(t,ninfo);

% perceptron parameters
P = ones([ninfo,1]);
w = 0;
alpha_P = 0.1;
% epochs_P = 100;

% initiate average cumulative rewards
R_proposed_cum_C = zeros([t,1]);
R_prfct2_cum = zeros([t,1]);
R_pred_cum = zeros([t,1]);
R_P_cum = zeros([t,1]);
R_T_cum = zeros([t,1]);
R_svm_cum = zeros([t,1]);
R_L_cum = zeros([t,1]);

% initiante acceptance ratio
global_max_numA_C = zeros(t,1);
numA_P = zeros(t,1);
numA_T = zeros(t,1);
numA_svm = zeros(t,1);
numA_L = zeros(t,1);
numA_prfct2 = zeros(t,1);
numA_pred = zeros(t,1);
ratioAs_C = zeros(t,1);
ratioAs_P = zeros(t,1);
ratioAs_T = zeros(t,1);
ratioAs_svm = zeros(t,1);
ratioAs_L = zeros(t,1);
ratioAs_prfct2 = zeros(t,1);
ratioAs_pred = zeros(t,1);

% initiate default probability
global_max_default_num_C = zeros([t,1]);
default_prob_C = zeros([t,1]);
default_num_P = zeros([t,1]);
default_prob_P = zeros([t,1]);
default_num_T = zeros([t,1]);
default_prob_T = zeros([t,1]);
default_num_svm = zeros([t,1]);
default_prob_svm = zeros([t,1]);
default_num_L = zeros([t,1]);
default_prob_L = zeros([t,1]);
default_num_prfct2 = zeros([t,1]);
default_prob_prfct2 = zeros([t,1]);
default_num_pred = zeros([t,1]);
default_prob_pred = zeros([t,1]);

%% iterations

for t_idx = 1:t
    
    % generate applicants number & info
    N = randi(Nlim,1);
%     N = 10;
    Nt(t_idx) = N; sum_Nt = sum(Nt);
    nempty = ceil(nempty_prcnt*N*ninfo);

    % picking data
    s_id = randi(numel(p_data),[N,1]);
    s = s_data(s_id,1:ninfo);
    p = p_data(s_id);
    LoanStatus = status_data(s_id);
    
    % assign group size
    if isgroup
        ngroup = s(:,1);
        s(:,1) = (4/100).*ngroup;
    else
        ngroup = ones(size(p));
    end
    
    % assign empty
    emty_idx = randperm(numel(s(:)),nempty);
    s(emty_idx) = NaN;

    % dealing with empty information for perceptron, SVM, and
    % Logistic Regression
    s_ne = s; 
    for feature_idx = 1:ninfo
        s_ne_temp = s_ne(:,feature_idx);
        s_ne_temp(isnan(s_ne_temp)) = mean(s_ne_temp,'omitnan');
        s_ne(:,feature_idx) = s_ne_temp;
    end
    s_ne(isnan(s_ne)) = 0;
    
    %% perfect decision constant limit
    
    % making decision
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
    
    % group reward
    R_prfct2 = R_prfct2.*ngroup;
    
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
%         s_temp = s; s_temp(isnan(s)) = 
        s_interp_pred = s_ne; 
        p_interp_pred = p;
        
        % initialize point search
        xs_pred = -10 + 20.*rand(10,numel(s(1,:))+1);
        loan_mdl_pred = @(x) sum((credit_score_model( ...
            s_interp_pred,x) - p_interp_pred).^2);
        fs = zeros(10,1);
        for xs_idx = 1:N
            [xs_pred(xs_idx,:),fs(xs_idx)] = ...
                fminunc(loan_mdl_pred,xs_pred(xs_idx,:),...
                optimoptions('fminunc','Display','none'));
        end
        xs_idx = find(fs==min(fs));
        xs_pred = xs_pred(xs_idx(1),:);

    else
        tic
        if t_idx <= train_lim
        
            % interpolation model
            loan_mdl_pred = ...
                @(x) sum(abs(credit_score_model(...
                s_interp_pred,x) - p_interp_pred));
            xs_pred = fminunc(loan_mdl_pred,xs_pred,...
                optimoptions('fminunc','Display','none'));
            % xs_pred = fminsearch(loan_mdl_pred,xs_pred,...
            %     optimset('Display','none'));

            % store data for next interpolation
            s_interp_pred = [s_interp_pred;s_ne];
            p_interp_pred = [p_interp_pred;p];
        
        end
        
        % making decision
        p_pred = credit_score_model(s_ne,xs_pred);
        A_pred = (p_pred >= ((1-e)/(1+c)));

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
        comp_time_pred = toc;
    end
    
    % group reward
    R_pred = R_pred.*ngroup;
    
    %% perceptron
    
    tic
    
    % calculate the decision value
    sP = s_ne./4;
    
    % making decision
    A_P = ((sP*P) + w) >= 0;

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
    for app_id = 1:size(sP,1)
        y_pred = (dot(P,sP(app_id,:)) + w) >= 0;
        % update weights and bias if prediction is incorrect
        if y_pred ~= LoanStatus(app_id)
            P = P + (alpha_P * (LoanStatus(app_id) - y_pred) * sP(app_id,:))';
            w = w + alpha_P * (LoanStatus(app_id) - y_pred);
        end
    end
    
    % group reward
    R_P = R_P.*ngroup;

    comp_time_P = toc;
    
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
        s_train_T = s_ne;
        dec_train_T = LoanStatus;
        
    else
        tic
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_T = fitctree(s_train_T,dec_train_T);

            % store data for training
            s_train_T = [s_train_T;s_ne];
            dec_train_T = [dec_train_T;LoanStatus];
        
        end
        
        % making decision
        A_T = predict(loan_mdl_T,s_ne);

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
        
        comp_time_FitTree = toc;
    end
    
    % group reward
    R_T = R_T.*ngroup;
    
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
        s_train_svm = s_ne;
        dec_train_svm = double(LoanStatus);
        
    else
        tic
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_svm = fitcsvm(s_train_svm,dec_train_svm);

            % store data for training
            s_train_svm = [s_train_svm;s_ne];
            dec_train_svm = [dec_train_svm;double(LoanStatus)];
        
        end
        
        % making decision
        A_svm = logical(predict(loan_mdl_svm,s_ne));

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
       
        comp_time_SVM = toc;
    end
    
    % group reward
    R_svm = R_svm.*ngroup;
    
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
        s_train_L = s_ne;
        dec_train_L = double(LoanStatus);
        
    else
        tic
        if t_idx <= train_lim
        
            % train a model
            loan_mdl_L = glmfit(s_train_L, dec_train_L, 'binomial');

            % store data for training
            s_train_L = [s_train_L;s_ne];
            dec_train_L = [dec_train_L;double(LoanStatus)];
        
        end
        
        % making decision
%         A_L = logical(predict(loan_mdl_L,s_ne));
        A_L = (0.5 < glmval(loan_mdl_L,s_ne,'logit'));

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
       
        comp_time_L = toc;
    end
    
    % group reward
    R_L = R_L.*ngroup;
        
    %% Proposed Approach
    
    % updating step size
    alpha_C = DG_C;%/sqrt(t_idx);

    if t_idx <= MSO_itr
        
        % gradient update for each particle
        R_sum_C = zeros(1,nPartcl);
        numA_C = zeros(1,nPartcl);
        default_num_C = zeros(1,nPartcl);
        for p_idx = 1:nPartcl
                        
            % current particle
            phi_now = phi_C(:,p_idx);
            eps_now = eps_C(:,p_idx);
            
            % perform gradient ascent
            [phi_C(:,p_idx),eps_C(:,p_idx),R_sum_C(p_idx),...
                numA_C(p_idx),default_num_C(p_idx)] = ...
                gradient_ascent(N,s,ngroup,ninfo,LoanStatus,k,...
                c,e,phi_now,eps_now,alpha_C,R_proposed_cum_C,sum_Nt);
            
        end
    
        % Pick the maximum					 
        global_max_R_C = max(R_sum_C); % update global maximum
		global_max_idx_C = find(R_sum_C == global_max_R_C);
		global_max_phi_C = phi_C(:,global_max_idx_C(1));
        global_max_eps_C = eps_C(:,global_max_idx_C(1));
        
        % calculate acceptance probability
        global_max_numA_C(t_idx) = numA_C(global_max_idx_C(1));
        ratioAs_C(t_idx) = sum(global_max_numA_C)/sum_Nt;
        
        % calculate default probability
        global_max_default_num_C(t_idx) = ...
            default_num_C(global_max_idx_C(1));
        default_prob_C(t_idx) = ...
		    sum(global_max_default_num_C)/sum(global_max_numA_C);
        
        % update parameters
        [~,srt_idx] = sort(R_sum_C,'descend');
        phi_C(:,1:nkeep) = phi_C(:,srt_idx(1:nkeep));
        phi_C(:,nkeep+1:end) = z_lb + z_rng.*rand(ninfo,nPartcl-nkeep);
        eps_C(:,1:nkeep) = eps_C(:,srt_idx(1:nkeep));
        eps_C(:,nkeep+1:end) = z_lb + z_rng.*rand(ninfo,nPartcl-nkeep);
		
    else
        
        tic
        % continue the parameters
        phi_now = global_max_phi_C;
        eps_now = global_max_eps_C;
        
        % perform gradient ascent
        [global_max_phi_C,global_max_eps_C,global_max_R_C,...
            global_max_numA_C(t_idx),global_max_default_num_C(t_idx)] = ...
            gradient_ascent(N,s,ngroup,ninfo,LoanStatus,k,...
            c,e,phi_now,eps_now,alpha_C,R_proposed_cum_C,sum_Nt);
        
        % calculate acceptance probability
        ratioAs_C(t_idx) = sum(global_max_numA_C)/sum_Nt;
        
        % calculate default probability
        default_prob_C(t_idx) = ...
            sum(global_max_default_num_C)/sum(global_max_numA_C);
        
        comp_time_C = toc;
        
    end
    
    % store maximum
    phis_C(t_idx,:) = global_max_phi_C';
    eps_arr_C(t_idx,:) = global_max_eps_C';
    
    %% cumulative rewards over time
    
    R_proposed_cum_C(t_idx) = global_max_R_C;
    R_prfct2_cum(t_idx) = sum(R_prfct2);
    R_pred_cum(t_idx) = sum(R_pred);
    R_P_cum(t_idx) = sum(R_P);
    R_T_cum(t_idx) = sum(R_T);
    R_svm_cum(t_idx) = sum(R_svm);
    R_L_cum(t_idx) = sum(R_L);

end

%% compile output

% compile cumulative rewards
R_cum.proposed.C = R_proposed_cum_C;
R_cum.perfect.constant = R_prfct2_cum;
R_cum.p_prediction = R_pred_cum;
R_cum.perceptron = R_P_cum;
R_cum.FitTree = R_T_cum;
R_cum.SVM = R_svm_cum;
R_cum.LogisticRegression = R_L_cum;

% compile acceptance ratio
acceptance.proposed.C = ratioAs_C;
acceptance.perfect.constant = ratioAs_prfct2;
acceptance.p_prediction = ratioAs_pred;
acceptance.perceptron = ratioAs_P;
acceptance.FitTree = ratioAs_T;
acceptance.SVM = ratioAs_svm;
acceptance.LogisticRegression = ratioAs_L;

% compile default probability
default.proposed.C = default_prob_C;
default.perfect.constant = default_prob_prfct2;
default.p_prediction = default_prob_pred;
default.perceptron = default_prob_P;
default.FitTree = default_prob_T;
default.SVM = default_prob_svm;
default.LogisticRegression = default_prob_L;

% compile parameters
parameters.proposed.C.phis = phis_C;
parameters.proposed.C.eps = eps_arr_C;
parameters.perceptron.p_prediction = loan_mdl_pred;
parameters.perceptron.P = P;
parameters.perceptron.w = w;
parameters.FitTree = loan_mdl_T;
parameters.SVM = loan_mdl_svm;
parameters.LogisticRegression = loan_mdl_L;

% comp_time.B = comp_time_B;
comp_time.C = comp_time_C;
comp_time.pred = comp_time_pred;
comp_time.P = comp_time_P;
comp_time.FitTree = comp_time_FitTree;
comp_time.SVM = comp_time_SVM;
comp_time.L = comp_time_L;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% function for the gradient ascent step

function [phi_now,eps_now,R_sum,numA,default_num] = ...
    gradient_ascent(N,s,ngroup,ninfo,LoanStatus,k,c,e,...
    phi_now,eps_now,alpha,R_proposed_cum,sum_Nt)

% calculate Q
s_phi = s.*phi_now';
Q = s_phi + eps_now';
    Qs = sign(Q); Q = abs(Q); Q(isinf(Q)) = realmax; Q = Qs.*Q;

% policy pi
exp_Q = exp(mean((k.*Q),2,'omitnan'));
    exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

% L form
pie = (2.*exp_Q./(1+exp_Q))-1;
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
del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,k);

% get acceptance and default numbers
numA = sum(A);
default_num = sum(R == (-1+e));

% group reward
R = R.*ngroup;

if (sum_Nt - numel(A)) == 0
    Rbar = 0;
else
    Rbar = sum(R_proposed_cum)/(sum_Nt - numel(A));
end
 
deltaR = (R - Rbar);
F = mean((del_pi./pie).*deltaR,1,'omitnan');
    Fs = sign(F); F = abs(F); F(isinf(F)) = realmax; F = Fs.*F;

% update parameters
z = [phi_now;eps_now];
z = z + alpha.*F';
    zs = sign(z); z = abs(z); z(isinf(z)) = realmax; z = zs.*z;
phi_now = z(1:ninfo);
eps_now = z(ninfo+1:end);

% summing rewards
R_sum = sum(R,'omitnan');

end

%% function to calcualte the derevative of the probability function

function del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,k)

del_pi = zeros([N, 2*ninfo]);

exp_Q = exp(k.*Q);
    exp_Q(exp_Q==0) = realmin; exp_Q(isinf(exp_Q)) = realmax;

exp_plus = exp_Q + 1;
exp_sqr = exp_plus.^2;
    exp_sqr(exp_sqr==0) = realmin; exp_sqr(isinf(exp_sqr)) = realmax;

par_del = 2.*exp_Q./exp_sqr;

del_pi(Aid,1:ninfo) = s(Aid,:).*par_del(Aid,:);
del_pi(Aid,ninfo+1:end) = par_del(Aid,:);
    del_pi_s = sign(del_pi); del_pi = abs(del_pi);
    del_pi(del_pi==0) = realmin; del_pi(isinf(del_pi)) = realmax;
    del_pi(isnan(del_pi)) = realmin; del_pi = del_pi_s.*del_pi;
    
end

%% credit score model

function p = credit_score_model(s,x)
    c = x(1:end-1)*s'+x(end);
    p = 1./(1+exp(-c'));
%     p = exp(c')./(1+exp(c'));
end



