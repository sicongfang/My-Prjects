function expectation = EPV(lambda,mu, c, u, b, i,delta)
    expectation = 0;
    t0 = 0;
    u0  = u;
    t=0;
    while t<1000
         dt = - log ( rand ( 1, 1 ) ) / lambda;
         t= t0+dt;
         u= funU(u0, c, dt);
         bt= funB(b, i, t);
         m= u-bt;  
         if m > 0 
             syms x float positive
             solfun = u0+c.*(x) == b.*exp(i.*(x+t0));
             solx = solve(solfun, x);
             solx = vpa(solx);
             fun = @(t) exp(-(delta.*t)) .* (u0+c.*(t-t0)-b.*exp(i.*(t))); %(funU(u0,c,t-t0) - funB(b,i,t));
             psum =integral(fun, double(solx+t0), double(t));
             expectation = double(expectation+psum); 
             u0 = bt-exprnd(mu);
             %fprintf('M>0: '); disp(expectation);
             
         else 
             syms x float positive
             solfun = u0+c.*(x) == b.*exp(i.*(x+t0));
             solx = solve(solfun, x);
             solx = solx(:);
             size1 = size(solx,2);
             if size1 == 2
                 fun = @(t) exp(-(delta.*t)) .* (u0+c.*(t-t0)-b.*exp(i.*(t)));
                 psum =integral(fun, double(t0+solx(1)), double(solx(2)+t0));
                 expectation = double(expectation+psum);
             end
             u0= u-exprnd(mu);
             %fprintf('M<0: '); disp(expectation);
             
         end
         if u<0 
             fprintf('ruin');
             disp(expectation);
             return
         end
         t0 = t;
    end
    disp(expectation);
end