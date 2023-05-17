

% initializing matrix, right-hand vector, and dimensions
A = [1 1 1; 1 1 1; 1 1 1;] % this is just a placeholder, user should replace with intended matrix
V = [0;0;0] % this is just a placeholder, user should replace with intended right-hand vector
A = cat(2, A, V) % concatenating matrix and vector
U = A
M = size(A)
 n = M(:, 1)
 m = M(:, 2)

currentcol = 1
currentrow = 1
% iterate through each column or each row
while currentrow <= n && currentcol <= m
    % if current column is all zeros, proceed to next column
	if A(:, currentcol) == zeros(n, 1)
		currentcol = currentcol+1
    else
        % sort each row including and below current row by absolute value of the pivot column entry
		C = A(1 : currentrow-1, :)
            B = A(currentrow : end, :)
            B = sortrows(B, currentcol, "descend",  'ComparisonMethod', 'abs')
            A = cat(1, C, B)
        
        % if pivot entry is still zero after
        % sort then move to next column while preserving row
        if A(currentrow, currentcol) == 0
            currentcol = currentcol + 1
            continue
        end

		% if B is all zeros, then GJ elimination is complete
		if B == zeros(size(B))
			break
        else
            % scaling the current row by the pivot entry
			A(currentrow,:) = A(currentrow,:)*1/A(currentrow, currentcol)
            
            % changing other entries in pivot column to zero
			for i = 1:n
				if i ~= currentrow
					A(i,:) = A(i,:) - A(i,currentcol)*A(currentrow,:)
				end
			end
        end
    % moving on to next pivot entry
	currentcol = currentcol + 1
	currentrow = currentrow + 1
	end
end

% displaying final RREF matrix
R = A

