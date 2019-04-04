function P = Pruin(lambda,mu, c, u, b, i)
    t0 = 0;
    u0  = u;
    t=0;
    while t<10000
        dt = - log ( rand ( 1, 1 ) ) / lambda;
        t= t0+dt;
        u= funU(u0, c, dt);
        bt= funB(b, i, t);
        m= u-bt;
        if m > 0
            u0= bt-exprnd(mu);
        else
            u0= u-exprnd(mu);
        end
        if u<0
            P=1;
            return
        end
        t0 = t;
    end
    P=0;
    return
end


