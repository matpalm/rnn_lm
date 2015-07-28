  # ./simple_rnn.py training test --adaptive-learning-rate=vanilla \
  #   | grep COST > vanilla.cost
  # ./simple_rnn.py training test \
  #   | grep COST > rmsprop.cost
  # ./bidirectiona_rnn.py training test \
  #   | grep COST > bidirectional.cost
  
  # 800x400 cost.png
  
  library(ggplot2)
  
  df1 = read.delim("vanilla.cost", h=F)
  df1$run = 'vanilla'
  df1$n = 1:nrow(df1)
  
  df2 = read.delim("rmsprop.cost", h=F)
  df2$run = 'unidirectional'
  df2$n = 1:nrow(df2)
  
  df3 = read.delim("bidirectional.cost", h=F)
  df3$run = 'bidirectional'
  df3$n = 1:nrow(df3)
  
  df4 = read.delim("gru.cost", h=F)
  df4$run = 'gru'
  df4$n = 1:nrow(df4)
  
  df = rbind(df1, df2, df3, df4)
  ggplot(df, aes(n, V2)) +
    geom_point(alpha=0.1, aes(colour=run)) +
    geom_smooth(aes(colour=run)) + ylim(0, 2) + 
    ggtitle("training cost convergence")
