function ruinprob = RP(lambda,mu, c, u)
    ruinprob = lambda*exp(u*(lambda/c-mu))/(c*mu);
end