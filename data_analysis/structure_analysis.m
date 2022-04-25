function data_structure = structure_analysis(data)
G_languages = groupcounts(data,'description_languages');
data_structure.G_languages = G_languages;

G_activity_sector = groupcounts(data,{'activity','sector'});
data_structure.G_activity_sector = G_activity_sector;

G_country_town = groupcounts(data,{'location_country','location_town'});
data_structure.G_country_town = G_country_town;

G_pictured = groupcounts(data,'borrowers_pictured');
data_structure.G_pictured = G_pictured;

G_gender = groupcounts(data,'borrowers_gender');
data_structure.G_gender = G_gender;

G_currency = groupcounts(data,'terms_disbursal_currency');
data_structure.G_currency = G_currency;

G_loss = groupcounts(data,'terms_loss_liability_nonpayment');
data_structure.G_loss = G_loss;

G_exchange = groupcounts(data,'terms_loss_liability_currency_exchange');
data_structure.G_exchange = G_exchange;

G_delinquent = groupcounts(data,'delinquent');
data_structure.G_delinquent = G_delinquent;

duration_nor = fitdist(data.duration,'Normal');
data_structure.duration_nor = duration_nor;

entries_nor = fitdist(data.journal_totals_entries,'Normal');
data_structure.entries_nor = entries_nor;

amount_nor = fitdist(data.terms_loan_amount,'Normal');
data_structure.amount_nor = amount_nor;
end