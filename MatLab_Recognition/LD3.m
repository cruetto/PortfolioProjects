%% 1.1
net = googlenet;

I = imread("peppers.png"); % skaito faila
inputSize = net.Layers(1).InputSize; % iesko failo resoliucija
I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
% nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija

[label,scores] = classify(net,I); % pasiima klases pavadinima ir kiek procentu
figure
imshow(I)
classNames = net.Layers(end).ClassNames; % Sugeneruoja masiva su klases pavadinimais
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");


%% 1.1

disp(numel(classNames)) % Visos klases
disp(classNames(1:15)) % Pirmos 15 klasiu


%% 1.2
net = googlenet;
inputSize = net.Layers(1).InputSize; % iesko failo resoliucija

for i=1:10
    I = imread(string(i) + ".jpg"); % skaito faila
    I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
    % nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija
    
    [label,scores] = classify(net,I);
    figure
    imshow(I)
    title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");
end

%% 1.3
net = googlenet;
inputSize = net.Layers(1).InputSize;
disp(inputSize)

for i=1:10
    I = imread(string(i) + ".jpg"); % skaito faila
    inputSize = net.Layers(1).InputSize; % iesko failo resoliucija
    I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
    % nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija
    
    [label,scores] = classify(net,I);

    [~,idx] = sort(scores,'descend');
    idx = idx(5:-1:1);
    classNamesTop = net.Layers(end).ClassNames(idx);
    scoresTop = scores(idx);
    
    figure
    barh(scoresTop)
    xlim([0 1])
    title(string(i)+" image")
    xlabel('Probability')
    yticklabels(classNamesTop)
end

%% 1.4

net = mobilenetv2();

inputSize = net.Layers(1).InputSize;
disp(inputSize)

for i=1:10
    I = imread(string(i) + ".jpg"); % skaito faila
    inputSize = net.Layers(1).InputSize; % iesko failo resoliucija
    I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
    % nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija
    
    [label,scores] = classify(net,I);
    figure
    imshow(I)
    title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");
end

%% 2 GoogleNet

net = googlenet();
inputSize = net.Layers(1).InputSize; % iesko failo resoliucija

for i=1:10
    I = imread(string(i) + ".jpg"); % skaito failus
    I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
    % nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija

    label = classify(net,I);

    scoreMap = gradCAM(net,I,label);
    figure
    imshow(I)
    hold on
    imagesc(scoreMap,'AlphaData',0.5)
    colorbar
    colormap jet
    title(string(label));
end

%% 2 MobileNetv2

net = mobilenetv2();
inputSize = net.Layers(1).InputSize; % iesko failo resoliucija

for i=1:10
    I = imread(string(i) + ".jpg"); % skaito failus
    I = imresize(I,inputSize(1:2)); % keicia rezoliucijos dydi,
    % nes googlenet gali identifikuoti klase tik su apibrezta rezoliucija

    label = classify(net,I);
    scoreMap = gradCAM(net,I,label);
    figure
    imshow(I)
    hold on
    imagesc(scoreMap,'AlphaData',0.5)
    colorbar
    colormap jet
    title(string(label))
end
