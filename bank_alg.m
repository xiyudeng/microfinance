function R_avg = bank_alg(N,t,c, ninfo, nempty)
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
end
