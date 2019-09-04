# win_size = 5
# overlap = 0
# distance = win_size - overlap
#
# results <- zoo::rollapply( data = roll_data,
#                 width = win_size,
#                 by = distance,
#                 FUN = sum) %>%  data.frame()
