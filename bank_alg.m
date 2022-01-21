% standard algorithm bank

function R_avg = bank_alg(N,t,c, ninfo)
    phat = 1;
    R_cum = zeros([t,1]);
    R_avg = zeros([t,1]);
    for i = 1:t
        %% get appliant infomation
        s = random_apc_info(N, ninfo, nempty);

        %% update the s_database and initial new entry
        if i == 1
            % s_database structure
            % col 1:ninfo: s infomation
            % col ninfo+1: prob = #return / # approve
            % col ninfo+2: # return
            % col ninfo+3: # approve
            s_database = [unique(s,'rows'), phat * ones(N,1), zeros(N,1), zeros((N,1))]; %initial all prob as phat
        else % i not 1
            if (~ismember(unique(s,'rows'), s_database, 'rows')) % new entry
                diff_s = setdiff(unique(s,'rows'),s_database); 
                s_database = [diff_s, phat*ones(length(diff_s),1)]; % set prob as initial phat
            end
        end

        %% get the action A
        % if prob >= 1/(1+c) : approve
        % if prob < 1/(1+c) : approve according to prob

        idx_s = find(s_database(:,1:ninfo) == s);
        p = s_database(idx_s,ninfo+1);

        A = zeros(N,1); % initial A list as 0
        x = rand; % a random number [0,1]
        A(p >= x) = 1; % apply 1 according to prob
        A(p >= (1/(1+c))) = 1; % case: prob >= 1/(1+c)
       
        %% calculate the return
        p_return = returned_prob(s, ninfo);

        return_varialbe = rand(N,1);
        R = zeros([N,1]);
        R(A == 1 & return_varialbe < p_return) = 1+c;
        R(A == 1 & return_varialbe >= p_return) = -1;

        R_cum(i) = sum(R);
        R_avg(i) = R_cum(i)/N;

        %% update the s_database with new prob
        if i == 1
            [~,~,ix] = unique(s,'rows');
            count = accumarray(ix,1).'; % count
        else
        end


        % df[df['score'] == 85]
        
    end
end
