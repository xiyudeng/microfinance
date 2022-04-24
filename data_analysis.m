%% load prprocessed kiva example data
kiva = readtable('paid_defaulted.csv');
kiva = removevars(kiva,{'Var1'});

%% categorical -> numerical; NaN -> 0
[description_languages, ~, G] = unique(kiva(:,{'description_languages'}));
kiva.description_languages = G-1; % make nan to 0

[activity, ~, G] = unique(kiva(:,{'activity'}));
kiva.activity = G;

[sector, ~, G] = unique(kiva(:,{'sector'}));
kiva.sector = G;

[location_country, ~, G] = unique(kiva(:,{'location_country'}));
kiva.location_country = G;

[location_town, ~, G] = unique(kiva(:,{'location_town'}));
kiva.location_town = G-1; % make nan to 0

[borrowers_gender, ~, G] = unique(kiva(:,{'borrowers_gender'}));
kiva.borrowers_gender = G;

[borrowers_pictured, ~, G] = unique(kiva(:,{'borrowers_pictured'}));
kiva.borrowers_pictured = G;

[borrowers_gender, ~, G] = unique(kiva(:,{'borrowers_gender'}));
kiva.borrowers_gender = G;

[terms_disbursal_currency, ~, G] = unique(kiva(:,{'terms_disbursal_currency'}));
kiva.terms_disbursal_currency = G;

[terms_loss_liability_nonpayment, ~, G] = unique(kiva(:,{'terms_loss_liability_nonpayment'}));
kiva.terms_loss_liability_nonpayment = G;

[terms_loss_liability_currency_exchange, ~, G] = unique(kiva(:,{'terms_loss_liability_currency_exchange'}));
kiva.terms_loss_liability_currency_exchange = G;

[delinquent, ~, G] = unique(kiva(:,{'delinquent'}));
kiva.delinquent = G;

kiva.duration(isnan(kiva.duration)) = 0;
