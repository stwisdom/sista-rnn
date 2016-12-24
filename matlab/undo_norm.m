function ynorm=undo_norm(y,std_cols,mean_ystd,mean_ymean)
    ynorm=y/std_cols;
    ynorm=ynorm*mean_ystd;
    ynorm=ynorm+mean_ymean;
    ynorm(ynorm<0.0)=0.0;
    ynorm(ynorm>1.0)=1.0;