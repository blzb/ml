function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================
admited_x = [];
admited_y =[];
notadmited_x = [];
notadmited_y = [];
for i =1: length(X)
    if(y(i) == 1)
        admited_x = [admited_x, X(i,1)];
        admited_y = [admited_y, X(i,2)];
    else
        notadmited_x = [notadmited_x, X(i,1)];
        notadmited_y = [notadmited_y, X(i,2)];
    end
    if(y(i) == 0)
    end        
end

plot(admited_x, admited_y, 'k+','MarkerSize', 7,'LineWidth', 2);
hold on
plot(notadmited_x, notadmited_y, 'ko','MarkerFaceColor', 'y','MarkerSize', 7);
hold off

end
