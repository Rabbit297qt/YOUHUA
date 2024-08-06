clc,clear,close all

% % 0000
% % 生成数据集
% rng(1);
% X = [randn(50,2)+ones(50,2);randn(50,2)-ones(50,2)];
% y = [-ones(50,1);ones(50,1)];
% 
% % 可视化数据
% figure;
% plot(X(y==1,1),X(y==1,2),'r+');
% hold on
% plot(X(y==-1,1),X(y==-1,2),'bo');
% hold off
% 
% % 选择核函数和参数
% kernel = 'rbf';
% C = 1;
% gamma = 10;
% 
% % 训练模型
% model = svmtrain(y,X,['-t ',num2str(2),' -c ',num2str(C),' -g ',num2str(gamma)]);
% 
% % 预测和评估
% [predicted_label, accuracy, decision_values] = svmpredict(y,X,model);
% confusionmat(y,predicted_label)



% %%  1111
% %设置两类不同数据
% A = [3,7;6,6;4,6;5,6.5];
% B = [1,2;3,5;7,3;3,4;6,2.7;4,3;2,7];
% C = [A;B];%两类数据合并
% 
% % 设置不同类别标签
% table = [true true true true false false false false false false false];
% D = nominal(table);
% 
% 
% % 数据集和标签
% sd=C;
% Y=D;
% 
% % 原始数据图像
% subplot(1,2,1)
% gscatter(sd(:,1),sd(:,2),Y,'rg','+*');
% 
% % SVM
% SVMModel=fitcsvm(sd,Y,'KernelFunction','linear');
% [lable,score]=predict(SVMModel,sd);
% 
% % 画图
% subplot(1,2,2)
% h = nan(3,1); 
% h(1:2) = gscatter(sd(:,1),sd(:,2),Y,'rg','+*'); 
% hold on
% h(3) =plot(sd(SVMModel.IsSupportVector,1),sd(SVMModel.IsSupportVector,2), 'ko');%画出支持向量
% 
% % 画出决策边界
% w=-SVMModel.Beta(1,1)/SVMModel.Beta(2,1);%斜率
% b=-SVMModel.Bias/SVMModel.Beta(2,1);%截距
% x_ = 0:0.01:10;
% y_ = w*x_+b;
% plot(x_,y_)
% hold on
% legend(h,{'-1','+1','Support Vectors'},'Location','Southeast');
% axis equal
% hold off



% %222222
% % 设置两类不同数据
% A = [0.9,1;0.8,1.8;0.79,1.7;0.7,3;0.8,3.9;0.9,4.5;1.2,5.7;1.6,5.6;2.5,6.1;2.9,5.8; 2.9,1;3.1,1.4;3.6,1.2;5,2;5.5,3.9;4.9,4.5;4.2,5.9;3.6,5.6;2.5,5.1;2.9,5.3];
% B = [2.5,3.1;3.5,2.6;4.5,3.2;3.5,2;2.4,2;3.5,2.5;4.3,3.7;2.6,2.8;2.4,3;3.6,3.1;4.4,3.3; 2.5,4.0;3.5,4.1;4.5,4.2;1.5,4;2.4,4;3.5,4.5;4.3,3.7;3.6,3.8;2.4,3.5;3.6,3.7;4.4,3.3];
% C = [A;B];%两类数据合并
% 
% % 设置不同类别标签
% table = [true true true true true true true true true true true true true true true true true true true true false false false false false false false false false false false false false false false false false false false false false false];
% D = nominal(table);
% 
% % 数据集和标签
% sd=C;
% Y=D;
% 
% %  原始数据图像
% subplot(1,2,1)
% gscatter(sd(:,1),sd(:,2),Y,'rg','+*');
% 
% %  原始数据图像
% subplot(1,2,1)
% gscatter(sd(:,1),sd(:,2),Y,'rg','+*');
% %  SVM
% SVMModel=fitcsvm(sd,Y,'BoxConstraint',10,'KernelFunction','rbf','KernelScale',2^0.5*2);%使用高斯核函数
% 
% % SVMModel=fitcsvm(sd,Y,'KernelFunction','rbf','OptimizeHyperparameters',{'BoxConstraint','KernelScale'},  'HyperparameterOptimizationOptions',struct('ShowPlots',false));%使用超参数优化
% 
% %  画图
% subplot(1,2,2)
% h = nan(3,1); 
% h(1:2) = gscatter(sd(:,1),sd(:,2),Y,'rg','+*'); 
% hold on
% h(3) = plot(sd(SVMModel.IsSupportVector,1),sd(SVMModel.IsSupportVector, 2),'ko'); %画出支持向量
% 
% % 画出决策边界
% hb = 0.2; 
% [X1,X2] = meshgrid(min(sd(:,1)):hb:max(sd(:,1)),min(sd(:,2)):hb:max(sd(:,2)));
% % 得到所有取点的矩阵
% [lable,score]=predict(SVMModel,[X1(:),X2(:)]);
% scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2));
% contour(X1,X2,scoreGrid,[0 0]);%绘制等高线
% hold on
% legend('-1','+1','Support Vectors','分界线');
% axis equal
% hold off



%%33333
% 
% % 生成两类数据
% dataClass1 = bsxfun(@plus, randn(100,2) * 0.75, [1 1]);  % 类别1
% dataClass2 = bsxfun(@plus, randn(100,2) * 0.75, [-1 -1]);  % 类别2
%  
% % 合并数据
% data = [dataClass1; dataClass2];
% groups = [ones(100,1); -ones(100,1)];
%  
% % 数据可视化
% figure; hold on;
% scatter(dataClass1(:,1), dataClass1(:,2), 'r');
% scatter(dataClass2(:,1), dataClass2(:,2), 'b');
% title('Training Data');
% xlabel('Feature 1');
% ylabel('Feature 2');
% legend('Class 1', 'Class 2');
% 
% % 训练SVM分类器
% SVMModel = fitcsvm(data, groups, 'KernelFunction', 'linear', 'BoxConstraint', 1);
%  
% % 可视化SVM边界
% figure;
% hgscatter = gscatter(data(:,1), data(:,2), groups, 'rb', 'xo');
% hold on;
% hsv = plot(SVMModel.SupportVectors(:,1), SVMModel.SupportVectors(:,2), 'ko', 'MarkerSize', 8);
% title('SVM with linear kernel');
% legend(hsv, 'Support Vectors');
% xlabel('Feature 1');
% ylabel('Feature 2');
% 
% % 新数据点
% newData = [0.5 0.5; -0.5 -0.5];
%  
% % 使用训练好的SVM模型进行预测
% label = predict(SVMModel, newData);
%  
% % 输出预测结果
% disp('Predicted class labels for the new data points:');
% disp(label);


%44444
% 载入Iris数据集
load fisheriris
data = meas;  % 数据的特征
groups = species;  % 数据的类别

% 将文本类别转换为数值类别
 [groups, groupNames] = grp2idx(groups);
%  [~, groupNames] = grp2idx(groups);


% rng(1);
% t = templateSVM('Standardize',true,'KernelFunction','gaussian');
% SVMModel = fitcecoc(data,groups,'Learners',t,'FitPosterior',true,...
%     'ClassNames',{'setosa','versicolor','virginica'},...
%     'Verbose',2);
% 训练一个多类SVM分类器
SVMModel = fitcecoc(data, groups);
% SVMModel = fitcecoc(data, groups, 'Learners', 'svm', 'ClassNames', groupNames, 'Verbose', 2);

% 交叉验证
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
disp(['Classification loss: ', num2str(classLoss)]);

% 新数据点
newData = [5.1 3.5 1.4 0.2; 6.7 3.0 5.2 2.3];
% 使用训练好的SVM模型进行预测
predictedLabels = predict(SVMModel, newData);
% 显示预测结果
disp('Predicted class labels for the new data points:');
for i = 1:length(predictedLabels)
      disp(['Data point ', num2str(i), ': ', char(groupNames(predictedLabels(i)))]);
end

% 
% %%55555
% % 假设已经加载了MNIST数据集，分为训练集和测试集
% % load('mnist.mat');  % 载入预处理的MNIST数据集
%   load mnist_dataset
%   
% % 简单的像素归一化，将图像数据缩放到[0,1]
% trainImages = double(trainImages) / 255;
% testImages = double(testImages) / 255;
%  
% % 将图像数据展平为向量
% trainImages = reshape(trainImages, size(trainImages, 1) * size(trainImages, 2), []);
% testImages = reshape(testImages, size(testImages, 1) * size(testImages, 2), []);
% 
% % 训练一个多类SVM分类器
% t = templateSVM('Standardize', true, 'KernelFunction', 'polynomial');
% SVMModel = fitcecoc(trainImages', trainLabels, 'Learners', t, 'Coding', 'onevsall', 'Verbose', 2);
% 
% % 预测测试集
% predictedLabels = predict(SVMModel, testImages');
%  
% % 计算并显示分类准确率
% accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
% disp(['Classification accuracy: ', num2str(accuracy * 100), '%']);
% 
% confMat = confusionmat(testLabels, predictedLabels);
% confchart(confMat);
% title('Confusion Matrix for MNIST Data');




% %%66666
% load fisheriris; % 加载鸢尾花数据集
% X = meas(:,3:4); % 选择后两个特征作为训练数据
% Y = species;    % 标签
% %  [Y, YNames] = grp2idx(Y);
% 
%  
% rng(1);
% t = templateSVM('Standardize',true,'KernelFunction','gaussian');
% model = fitcecoc(X,Y,'Learners',t,'FitPosterior',true,...
%     'ClassNames',{'setosa','versicolor','virginica'},...
%     'Verbose',2);
% 
% % model = fitcecoc(X, Y, 'KernelFunction', 'linear'); % 使用线性核函数训练SVM模型
% % model = fitcecoc(X, Y, 'Learners', 'svm', 'ClassNames', YNames, 'Verbose', 2);
% % model = fitcecoc(X,Y);
% 
% % 对新数据进行预测
% new_data = [1.15, 0.2];
% predicted_label = predict(model, new_data);
% disp(predicted_label); % 输出预测结果




