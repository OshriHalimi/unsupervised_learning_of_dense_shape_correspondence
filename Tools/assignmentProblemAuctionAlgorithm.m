% 
% Implementation of Bertsekas' auction algorithm [1] for a very fast 
% solution of the linear assignment problem, i.e. it is able 
% to solve a 1000-by-1000 problem in ~1.5s.
%
% In practice, with respect to running time, the auction algorithm 
% outperforms the Kuhn-Munkres (or Hungarian) algorithm significantly. 
% The auction algorithm has average run-time complexity O(N^2*log(N)) and 
% worst-case complexity O(N^3), whereas the Kuhn-Munkres algorithm has 
% average- and worst-case complexity  O(N^3).
%
% Note that only global optima are found for integer-valued benefit matrices. 
% For real-valued benefit matrices a scaling of the values needs to be applied
% by multiplication with a large number. This scaling factor depends on the
% desired accuracy, as the global solution is found for the integral part
% of the benefit matrix, whilst there is no guarantee that the fractional part 
% of the benefits are properly taken into account. However, in practical cases 
% it seems advantageous to not round the resulting benefit matrix and retain 
% the fractional parts in the benefit matrix. Also note that larger scalings 
% of the benefit matrix increase the run-time, so a problem-specific trade-off 
% between runtime and accuracy must be chosen.
%
% Runtime can be slightly decreased by installing the FastSet library,
% available on 
% http://www.levmuchnik.net/Content/ProgrammingTips/MatLab/FastSet/FastSet.html
%
% Input:
%               A                       N-by-N benefit matrix (higher
%                                       values indicate a better match)
%               [epsilon]               Initial value of epsilon (optional)
%               [epsilonDecreaseFactor] Decreas factor of epsilon
%                                       (optional)
%
% Output:
%               assignments             Resulting assignments
%               [prices]                Prices used during auctions
%                                       (optional)
%
% Example:      See the function test() below for a usage example.
%               Typically only the benefit matrix A is given as input and the
%               first output argument is relevant. epsilon and
%               epsilonDecreaseFactor can be used to heuristically adapt
%               runtime. The example below can also be used for testing
%               whether a sufficient amount of scaling has been performed.
%               This is done by solving the assignment problem using CVX
%               (available on http://cvxr.com/cvx/download/ ) and comparing
%               its result to the solution with the auction algorithm.
%           
%
% [1]	Bertsekas, D.P. 1998. Network Optimization Continuous and Discrete Models.
%
% Implementation by Florian Bernard ( f.bernardpi [at] gmail [dot] com )
%
% Created on 25/07/2014

function [assignments, prices] = ...
    assignmentProblemAuctionAlgorithm(A, epsilon, epsilonDecreaseFactor)

N = size(A,1);
assignments = nan(N,1);
prices = ones(N,1);

% check which unique function we use
if ( exist('fast_unique', 'file') ) 
    % requires FastSet, available on 
    % http://www.levmuchnik.net/Content/ProgrammingTips/MatLab/FastSet/FastSet.html
    uniqueFcn = @fast_unique;
else
    uniqueFcn = @unique;
end

% heuristic for setting epsilon
A = A*(N+1);
if ( ~exist('epsilon', 'var') || isempty(epsilon) )
    maxAbsA = max(abs(A(:)));
    epsilon = maxAbsA/25;
%     if ( sum(abs(A(:)) > 0.0001) > 0.7*N*N ) % non-sparse
%         ...
%     else
%         epsilon = 0.5*((N*maxVal)/5 + N*maxVal); % see page 260 in [1]
%     end
end

if ( ~exist('epsilonDecreaseFactor', 'var') )
    epsilonDecreaseFactor = 0.2;
end

while (epsilon >= 1) 
    % the outer loop performs epsilon-scaling in order to speed up execution
    % time. In particular, an updated prices array is computed in each
    % round, which speeds up further rounds with lower values of epsilon.
    assignments = nan(N,1);
    while (any(isnan(assignments)))
    %% forward-auction algorithm
        %% bidding phase
        % find unassigned indices
        unassignedIdx = find(isnan(assignments));
        nUnassigned = numel(unassignedIdx);
        
        % find best and second best objects
        AijMinusPj = bsxfun(@minus, A(unassignedIdx,:), prices');

        [~,viIdx] = max(AijMinusPj, [], 2);

        for i=1:nUnassigned   
            AijMinusPj(i,viIdx(i)) = -inf;
        end
        wi = max(AijMinusPj, [], 2);
        
        % compute bids
        bids = nan(nUnassigned,1);
        for i=1:nUnassigned  
            bids(i) = A(unassignedIdx(i),viIdx(i)) - wi(i) + epsilon;
        end
       
        
        %% assignment phase
        objectsThatHaveBeenBiddedFor = uniqueFcn(viIdx);
        for uniqueObjIdx=1:numel(objectsThatHaveBeenBiddedFor)
            currObject = objectsThatHaveBeenBiddedFor(uniqueObjIdx);
            personssWhoGaveBidsForJ = find(viIdx==currObject);
            
            [prices(currObject), idx] = max(bids(personssWhoGaveBidsForJ));
            personWithHighestBid = unassignedIdx(personssWhoGaveBidsForJ(idx));
            
            % remove previous assignment and store new assignment (the person
            % with highest bid)
            assignments(assignments==currObject) = nan;
            assignments(personWithHighestBid) = currObject;
            
        end
    end
    epsilon = epsilon * epsilonDecreaseFactor; % refine epsilon
end
end

function test()
N = 1000;

X = rand(N,3);
reference = rand(N,3);

A = X*reference';
%% Apply Auction algorithm
tic
    % scale A such that the integer version of A has sufficient accuracy
    % if S1 and P are not equal, the scaling factor needs to be increased
    scalingFactor = 10^6;
    Ascaled = A*scalingFactor;

    [assignments] = assignmentProblemAuctionAlgorithm(Ascaled);

    % create permutation matrix from assignments
    linIdx = sub2ind(size(A),assignments',1:size(A,1));
    P = sparse(size(A,1),size(A,2));
    P(linIdx) = 1;
toc

%% Use CVX for checking correctness of result
%  (Useful in particular for non-integer matrices A)
%  Note that this is really slow as CVX needs to create the canonical form
%  of the LP before it can call Gurobi. Creating the canonical form by hand
%  and directly calling Gurobi is much faster.
E = ones(N,1);

tic
    cvx_begin % quiet
        cvx_solver gurobi % uncomment this line if Gurobi is not available
        variable S1(N,N) nonnegative;
        maximize (trace(S1*A));
        S1*E==E;
        E'*S1==E';
    cvx_end
toc

% check if results are equal
if ( sum(sum(full(S1)~=full(P))) == 0 )
    disp('Correct results using Auction algorithm');
else
    error('Incorrect results using Auction algorithm. You probably need to scale A by a larger factor!');
end
end
