function data_pool = pool_generater(pool_size, data_structure)
indicator = rand(pool_size,9)*100;

Percent = [0;cumsum(data_structure.G_languages.Percent)];
% Percent = [0 80 82 83 88 92 100];
languages = discretize(indicator(:,1),Percent);
data_languages = cell2table(data_structure.G_languages.description_languages(languages),"VariableNames",["description_languages"]);

Percent = [0;cumsum(data_structure.G_activity_sector.Percent)];
activity_sector = discretize(indicator(:,2),Percent);
data_activity = cell2table(data_structure.G_activity_sector.activity(activity_sector),"VariableNames",["activity"]);
data_sector = cell2table(data_structure.G_activity_sector.sector(activity_sector),"VariableNames",["sector"]);

Percent = [0;cumsum(data_structure.G_country_town.Percent)];

country_town = discretize(indicator(:,3),Percent);
data_country = cell2table(data_structure.G_country_town.location_country(country_town),"VariableNames",["location_country"]);
data_town = cell2table(data_structure.G_country_town.location_town(country_town),"VariableNames",["location_town"]);

Percent = [0;cumsum(data_structure.G_pictured.Percent)];
pictured = discretize(indicator(:,4),Percent);
data_pictured = cell2table(data_structure.G_pictured.borrowers_pictured(pictured),"VariableNames",["borrowers_pictured"]);

Percent = [0;cumsum(data_structure.G_gender.Percent)];
gender = discretize(indicator(:,5),Percent);
data_gender= cell2table(data_structure.G_gender.borrowers_gender(gender),"VariableNames",["borrowers_gender"]);

Percent = [0;cumsum(data_structure.G_currency.Percent)];
currency = discretize(indicator(:,6),Percent);
data_currency= cell2table(data_structure.G_currency.terms_disbursal_currency(currency),"VariableNames",["terms_disbursal_currency"]);

Percent = [0;cumsum(data_structure.G_loss.Percent)];
loss = discretize(indicator(:,7),Percent);
data_loss= cell2table(data_structure.G_loss.terms_loss_liability_nonpayment(loss),"VariableNames",["terms_loss_liability_nonpayment"]);

Percent = [0;cumsum(data_structure.G_exchange.Percent)];
% disp(Percent);
% Percent = [0, 30, 70, 100];
exchange = discretize(indicator(:,8),Percent);
data_exchange= cell2table(data_structure.G_exchange.terms_loss_liability_currency_exchange(exchange),"VariableNames",["terms_loss_liability_currency_exchange"]);

Percent = [0;cumsum(data_structure.G_delinquent.Percent)];
% Percent = [0, 50, 100];
delinquent = discretize(indicator(:,9),Percent);
data_delinquent= cell2table(data_structure.G_delinquent.delinquent(delinquent),"VariableNames",["delinquent"]);

data_duration = array2table(normrnd(data_structure.duration_nor.mu-150, data_structure.duration_nor.sigma, pool_size,1),"VariableNames",["duration"]);

data_entries = array2table(normrnd(data_structure.entries_nor.mu+2, data_structure.entries_nor.sigma+2, pool_size,1),"VariableNames",["journal_totals_entries"]);

data_amount = array2table(normrnd(data_structure.amount_nor.mu-300, data_structure.amount_nor.sigma+300, pool_size,1),"VariableNames",["terms_loan_amount"]);

data_pool = [data_languages, data_activity, data_sector, data_country, ...
    data_town, data_gender, data_pictured, data_currency, data_loss, data_exchange, data_delinquent,data_entries,data_amount,data_duration];

end
