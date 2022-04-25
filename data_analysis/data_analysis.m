%% load prprocessed kiva example data
kiva = readtable('paid_defaulted.csv');
kiva = removevars(kiva,{'Var1'});
kiva = preprocessing(kiva);