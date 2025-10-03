function newMean=UpdateMean(OldMean,NewDataValue,n)
newMean = (n*OldMean+NewDataValue)/(n+1);
end

function newStd=UpdateStd(OldMean,OldStd,NewMean,NewDataValue,n)
    newStd=sqrt(((OldStd^2)*(n-1)+NewDataValue^2+n*(OldMean^2)-(n+1)*(NewMean^2))/n);
end

function newMedian = UpdateMedian(OldMedian, NewDataValue, A, n)
    if (~mod(n, 2))  
        if (NewDataValue <A(n/2))
            newMedian=A(n/2);
        elseif (NewDataValue >A(n/2 + 1))
            newMedian=A(n/2+1);
        elseif (NewDataValue>A(n/2) && NewDataValue<A(n/2+1))
            newMedian=NewDataValue;
        end
    else  
        if (NewDataValue<A((n-1)/2))
            newMedian=(A((n-1)/2)+A((n+1)/2))/2;
        elseif (NewDataValue>A((n+3)/2))
            newMedian=(A((n+1)/2)+A((n+3)/2))/2;
        else
            newMedian=(A((n+1)/2)+NewDataValue)/2;
        end
    end
end

