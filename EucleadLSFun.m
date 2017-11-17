function [ deltaDist ] = EucleadLSFun( f, f1, f2 )
deltaDist = sum((f - f1).^2, 2) - sum((f - f2).^2, 2);
end

