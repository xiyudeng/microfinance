%% load prprocessed kiva example data
kiva = readtable('paid_defaulted.csv');
kiva = removevars(kiva,{'Var1'});

%% categorical -> numerical; NaN -> 0
% description_languages: categorical, one-hot encoding
a = kiva(:,{'description_languages'});
a = categorical(table2array(a));
languages = onehotencode(a,2);
kiva = addvars(kiva,languages,'Before','status');
kiva = removevars(kiva,{'description_languages'});

% activity: one-hot
a = kiva(:,{'activity'});
a = categorical(table2array(a));
activity = onehotencode(a,2);
kiva = addvars(kiva,activity,'Before','status');
kiva = removevars(kiva,{'activity'});

% sector: one-hot
a = kiva(:,{'sector'});
a = categorical(table2array(a));
sector = onehotencode(a,2);
kiva = addvars(kiva,sector,'Before','status');
kiva = removevars(kiva,{'sector'});


% country: one-hot
a = kiva(:,{'location_country'});
a = categorical(table2array(a));
country = onehotencode(a,2);
kiva = addvars(kiva,country,'Before','status');
kiva = removevars(kiva,{'location_country'});

% town: one-hot
a = kiva(:,{'location_town'});
a = categorical(table2array(a));
town = onehotencode(a,2);
kiva = addvars(kiva,town,'Before','status');
kiva = removevars(kiva,{'location_town'});

% pictured: one-hot
a = kiva(:,{'borrowers_pictured'});
a = categorical(table2array(a));
pictured = onehotencode(a,2);
kiva = addvars(kiva,pictured,'Before','status');
kiva = removevars(kiva,{'borrowers_pictured'});

% gender: one-hot
a = kiva(:,{'borrowers_gender'});
a = categorical(table2array(a));
gender = onehotencode(a,2);
kiva = addvars(kiva,gender,'Before','status');
kiva = removevars(kiva,{'borrowers_gender'});

% currency: one-hot
a = kiva(:,{'terms_disbursal_currency'});
a = categorical(table2array(a));
currency = onehotencode(a,2);
kiva = addvars(kiva,currency,'Before','status');
kiva = removevars(kiva,{'terms_disbursal_currency'});

% loss: one-hot
a = kiva(:,{'terms_loss_liability_nonpayment'});
a = categorical(table2array(a));
loss = onehotencode(a,2);
kiva = addvars(kiva,loss,'Before','status');
kiva = removevars(kiva,{'terms_loss_liability_nonpayment'});

% exchange: one-hot
a = kiva(:,{'terms_loss_liability_currency_exchange'});
a = categorical(table2array(a));
exchange = onehotencode(a,2);
kiva = addvars(kiva,exchange,'Before','status');
kiva = removevars(kiva,{'terms_loss_liability_currency_exchange'});

% delinquent: one-hot
a = kiva(:,{'delinquent'});
a = categorical(table2array(a));
delinquent = onehotencode(a,2);
kiva = addvars(kiva,delinquent,'Before','status');
kiva = removevars(kiva,{'delinquent'});

% kiva.duration(isnan(kiva.duration)) = 0;

% label paid/defaulted
[status, ~, G] = unique(kiva(:,{'status'}));
kiva.status = G-1; %0 defaulted 1 paid

% last column: label
kiva = movevars(kiva,'status','After','duration');
