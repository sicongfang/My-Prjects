function expectation = lin(mu, c, u, b, k,delta)

expectation = 0;
    t0 = 0;
    u0  = u;
    t=0;
    while t<50000
         dt = gamrnd(2.5,1);
         fprintf('dt');
         disp(dt);
         t= t0+dt;
        
         u = u0+c*dt; 
         bt = b+k*t;   
       fprintf('u');
         disp(u);
        fprintf('bt');
        disp(bt);
         m= u-bt;  
         if m > 0 
             solx=double(b-u0+c*t0)/(c-k);
             
             fun = @(x) exp(-(delta.*x)) .* (u0+c.*(x-t0)-(b+(k.*x))); %(funU(u0,c,t-t0) - funB(b,i,t));
             psum =integral(fun, double(solx), double(t));
             fprintf('psum');
             disp(psum);
             expectation = double(expectation+psum); 
             u0 = bt-exprnd(mu);
         end
         
         if u<0 
             fprintf('ruin');
             return;
         end
         t0 = t;
       %disp('expctation\n');
       %disp(expectation);
    end
return;

end