% Loading in the real images
I_3D = imread("/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Sample Images/000Images/000_1_00000.jpg");

% Have to load in the mask data as a csv, then do some processing
M_data = readmatrix("/Users/tyloftin/Downloads/000_Images00000.csv", 'Delimiter',';');


% Force into a 2D array
[rows, cols, channels] = size(I_3D);
if channels == 3
       I_gray = rgb2gray(I_3D);
   else
       I_gray = I_3D;
end

% Have to binarize image
I = imbinarize(I_gray);

% Grab cooridnates of mask
x = M_data(:,1);
y = M_data(:,2);

width = max(x) - min(x);
height = max(y) - min(y);

area = width * height;

mask_coords = [min(x) min(y);
               max(x) min(y);
               max(x) max(y);
               min(x) max(y);
               min(x) min(y)];

% Number of points per edge
nEdgePts = 200;

% Build dense boundary for mask
B_mask = [];
for k = 1:size(mask_coords,1)-1
    x_line = linspace(mask_coords(k,1), mask_coords(k+1,1), nEdgePts);
    y_line = linspace(mask_coords(k,2), mask_coords(k+1,2), nEdgePts);
    B_mask = [B_mask; [x_line(:), y_line(:)]];
end

% Setting the boundaries of the sample
B_real = bwboundaries(I);
B_real = B_real{1};

% Distance from each mask point to closest real box point
D = pdist2(B_mask, B_real);
minDist = min(D,[],2)    
meanDist = mean(minDist) 
maxDist  = max(minDist) 

% Fit percentage calculation
intersection = BW & M;
union_area   = BW | M;

fit_percentage = 100 * sum(intersection(:)) / sum(union_area(:))
