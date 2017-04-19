function sum = Loopsum(lambda,mu, c, u, b, i)
sum = 0;
for n = 1:1000
    sum = sum + Pruin(lambda,mu, c, u, b, i);
end
sum = sum/1000.00;
return;
end