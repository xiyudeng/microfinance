%% step 3: generate data
% feel free to change
desired_num = 20000;
% mean deseired defaulted 0.3, add uncertainty
desired_defaulted_rate = normrnd(0.3,0.1);

defaulted_num = round(desired_defaulted_rate * desired_num);
paid_num = desired_num - defaulted_num;

%% learn the distribution
kiva_original = readtable('paid_defaulted.csv');
kiva_original = removevars(kiva_original,{'Var1'});
idx = strcmp(kiva_original.status,'paid');
paid = kiva_original(idx,:);
defaulted = kiva_original(~idx,:);

pool_size = 10000;

%% learn paid/defaulted data structure
paid_structure = structure_analysis(paid);
defaulted_structure = structure_analysis(defaulted);

%% get paid/defaulted data pool
paid_pool = pool_generater(pool_size, paid_structure);
defaulted_pool = pool_generater(pool_size, defaulted_structure);

% paid_pool = preprocessing(paid_pool);
% defaulted_pool = preprocessing(defaulted_pool);

new = preprocessing([kiva_original(:,1:14);paid_pool]);
paid_pool = new(height(kiva_original)+1:end,:);
paid_pool = fillmissing(paid_pool,'constant',-1);
paid_pool_label = net(table2array(paid_pool)');
paid_idx = find(paid_pool_label==1);
paid_pool = paid_pool(paid_idx,:);
paid_pool_label = paid_pool_label(paid_idx);

new = preprocessing([kiva_original(:,1:14);defaulted_pool]);
defaulted_pool = new(height(kiva_original)+1:end,:);
defaulted_pool = fillmissing(defaulted_pool,'constant',-1);
defaulted_pool_label = net(table2array(defaulted_pool)');
defaulted_idx = find(defaulted_pool_label==0);
defaulted_pool = defaulted_pool(defaulted_idx,:);
defaulted_pool_label = defaulted_pool_label(defaulted_idx);


%% get label to paid/defaulted data pool

% indicator = rand(pool_num,9)*100;
% new_data = array2table(zeros(pool_num,0));
% 
% data_structure = struct();
% %%
% G_languages = groupcounts(paid,'description_languages');
% data_structure.G_languages = G_languages;
% Percent = [0;cumsum(G_languages.Percent)];
% languages = discretize(indicator(:,1),Percent);
% data_languages = cell2table(G_languages.description_languages(languages),"VariableNames",["description_languages"]);
% 
% %%
% G_activity_sector = groupcounts(paid,{'activity','sector'});
% data_structure.G_activity_sector = G_activity_sector;
% Percent = [0;cumsum(G_activity_sector.Percent)];
% activity_sector = discretize(indicator(:,2),Percent);
% data_activity = cell2table(G_activity_sector.activity(activity_sector),"VariableNames",["activity"]);
% data_sector = cell2table(G_activity_sector.sector(activity_sector),"VariableNames",["sector"]);
% 
% %%
% G_country_town = groupcounts(paid,{'location_country','location_town'});
% data_structure.G_country_town = G_country_town;
% Percent = [0;cumsum(G_country_town.Percent)];
% country_town = discretize(indicator(:,3),Percent);
% data_country = cell2table(G_country_town.location_country(country_town),"VariableNames",["location_country"]);
% data_town = cell2table(G_country_town.location_town(country_town),"VariableNames",["location_town"]);
% 
% %%
% G_pictured = groupcounts(paid,'borrowers_pictured');
% data_structure.G_pictured = G_pictured;
% Percent = [0;cumsum(G_pictured.Percent)];
% pictured = discretize(indicator(:,4),Percent);
% data_pictured = cell2table(G_pictured.borrowers_pictured(pictured),"VariableNames",["borrowers_pictured"]);
% 
% G_gender = groupcounts(paid,'borrowers_gender');
% data_structure.G_gender = G_gender;
% Percent = [0;cumsum(G_gender.Percent)];
% gender = discretize(indicator(:,5),Percent);
% data_gender= cell2table(G_gender.borrowers_gender(gender),"VariableNames",["borrowers_gender"]);
% 
% G_currency = groupcounts(paid,'terms_disbursal_currency');
% data_structure.G_currency = G_currency;
% Percent = [0;cumsum(G_currency.Percent)];
% currency = discretize(indicator(:,6),Percent);
% data_currency= cell2table(G_currency.terms_disbursal_currency(currency),"VariableNames",["terms_disbursal_currency"]);
% 
% 
% G_loss = groupcounts(paid,'terms_loss_liability_nonpayment');
% data_structure.G_loss = G_loss;
% Percent = [0;cumsum(G_loss.Percent)];
% loss = discretize(indicator(:,7),Percent);
% data_loss= cell2table(G_loss.terms_loss_liability_nonpayment(loss),"VariableNames",["terms_loss_liability_nonpayment"]);
% 
% G_exchange = groupcounts(paid,'terms_loss_liability_currency_exchange');
% data_structure.G_exchange = G_exchange;
% Percent = [0;cumsum(G_exchange.Percent)];
% exchange = discretize(indicator(:,8),Percent);
% data_exchange= cell2table(G_exchange.terms_loss_liability_currency_exchange(exchange),"VariableNames",["terms_loss_liability_currency_exchange"]);
% 
% G_delinquent = groupcounts(paid,'delinquent');
% data_structure.G_delinquent = G_delinquent;
% Percent = [0;cumsum(G_delinquent.Percent)];
% delinquent = discretize(indicator(:,9),Percent);
% data_delinquent= cell2table(G_delinquent.delinquent(delinquent),"VariableNames",["delinquent"]);
% 
% %% duration
% % h_duration = histfit(paid.duration,[ ],'Kernel');
% % ker = fitdist(paid.duration,'Kernel');
% duration_nor = fitdist(paid.duration,'Normal');
% data_structure.duration_nor = duration_nor;
% data_duration = array2table(normrnd(duration_nor.mu, duration_nor.sigma, pool_num,1),"VariableNames",["duration"]);
% 
% 
% %% journal_totals_entries
% % h_duration = histfit(paid.duration,[ ],'Kernel');
% % ker = fitdist(paid.duration,'Kernel');
% entries_nor = fitdist(paid.journal_totals_entries,'Normal');
% data_structure.entries_nor = entries_nor;
% data_entries = array2table(normrnd(entries_nor.mu, entries_nor.sigma, pool_num,1),"VariableNames",["journal_totals_entries"]);
% 
% %% terms_loan_amount
% % h_duration = histfit(paid.duration,[ ],'Kernel');
% % ker = fitdist(paid.duration,'Kernel');
% amount_nor = fitdist(paid.terms_loan_amount,'Normal');
% data_structure.amount_nor = amount_nor;
% data_amount = array2table(normrnd(amount_nor.mu, amount_nor.sigma, pool_num,1),"VariableNames",["terms_loan_amount"]);
% 
% %% 
% data_pool = [data_languages, data_activity, data_sector, data_country, ...
%     data_town, data_gender, data_pictured, data_currency, data_loss, data_exchange, data_delinquent,data_entries,data_amount,data_duration];