clc, clear
% image = rgb2gray(imread('shapes/circles/drawing(10).png'));
% normImage = 2 * mat2gray(image) - 1;
% result = normImage(:)';
% 
% result2 = [result; result]

% ==================== Training data ===========================
X = [];
y = [];

for i = 1:80
    textFileName = ['shapes/circles/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    X = [X; result];
end

for i = 1:80
    textFileName = ['shapes/squares/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    X = [X; result];
end

for i = 1:80
    textFileName = ['shapes/triangles/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    X = [X; result];
end

for i = 1:240
    if i > 160
        y = [y 3];
        continue
    end
    if i > 80
        y = [y 2];
        continue
    end
    y = [y 1];
end


% save('shapes/trainingData.mat',trainingData,label)

% ==================== Initial param theta1 and theta2
Theta1 = zeros(25, 785);

Theta2 = zeros(3, 26);

y = y';

save 'shapes/trainingData.mat' X y Theta1 Theta2

% ==================== Tesing data ===========================
testingData = [];
testingLabel = [];

for i = 81:100
    textFileName = ['shapes/circles/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    testingData = [testingData; result];
end

for i = 81:100
    textFileName = ['shapes/squares/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    testingData = [testingData; result];
end

for i = 81:100
    textFileName = ['shapes/triangles/drawing(' num2str(i) ').png'];
    image = rgb2gray(imread(textFileName));
    normImage = 2 * mat2gray(image) - 1;
    result = normImage(:)';
    testingData = [testingData; result];
end

for i = 1:60
    if i > 40
        testingLabel = [testingLabel 3];
        continue
    end
    if i > 20
        testingLabel = [testingLabel 2];
        continue
    end
    testingLabel = [testingLabel 1];
end

testingLabel = testingLabel';

save 'shapes/testingData.mat' testingData testingLabel