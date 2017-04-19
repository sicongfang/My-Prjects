function epv = REPV(lambda,mu, c, u, b, delta)
    ro = (lambda+delta-c*mu + sqrt((c*mu-delta-lambda)^2 + 4*c*mu*delta))/(2*c) ;
    R = (-lambda-delta+c*mu + sqrt((c*mu-delta-lambda)^2 + 4*c*mu*delta))/(2*c) ;
    epv = ((mu+ro)*exp(ro*u)-(mu-R)*exp(-R*u))/(ro*(mu+ro)*exp(ro*b)+R*(mu-R)*exp(-R*b));
end