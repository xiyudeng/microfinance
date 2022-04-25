% data generation method 1
% after preprocessing
% 3182 x 3165: last 4 numerical
% fill missing with -1
kiva = fillmissing(kiva,'constant',-1);
% data-label
data = removevars(kiva,{'status'});
label = kiva(:,{'status'});
x = table2array(data)';
y = table2array(label)';

%% step 1: run perceptron to find the threshold
net = perceptron;
% error weighted for unbalanced data: n_sample/(n_classes * n_samplesj)
% -> paid: 3182/(2*3082) = 0.516   -> defaulted 3182/(2*100)= 15.91
% train about 2000 epoches
net.trainParam.epochs = 200;
net.performFcn = 'mae';
net = train(net,x_train,y_train,[],[],[15.91,0.516]');
ybar = net(x_test);
test_error = sum(abs(y_test - ybar))/length(ybar);
disp('Test error:');
disp(test_error);
% ybar = net(x_test)
% net2 = configure(net,data,label);
wb = getwb(net);
% b: bias vectors	IW: input weight matrices	
[b,IW,~] = separatewb(net,wb);
b = b{1,1};
w = IW{1,1};


%% step 2: modify sigmoid



