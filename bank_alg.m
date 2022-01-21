% standard algorithm bank

function cumR = bank_alg(N,t,c, ninfo)
    phat = 1;
    for i = 1:t
        if i == 1
            bank_p = phat * ones(N,1); % phat
        end
        s = random_apc_info(N, ninfo, nempty);
        if (~ismember(unique(s,'rows'), s_database, 'rows'))
            diff_s = setdiff(unique(s,'rows'),s_database);
            s_database = [diff_s, phat*ones(length(diff_s),1)];
        end
        idx_s = find(s_database = s);
        p = s_database(idx_s, ninfo+1);
        A = zeros(N,1);
        A(p > 1/(1+c)) = 1;

        
        
    end
end
