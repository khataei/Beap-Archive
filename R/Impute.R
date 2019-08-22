#' Title
#'
#' @param data The input dataset, must have x_axis, y_axis, z_axis columns
#' @param time_period_length The time period after of before the NA value
#' @param k Number of samples to be considered for NA imputation
#'
#' @return main_df Imputed results
#'
#' @import magrittr
#' @importFrom lubridate seconds
#' @export
#'
#'
Impute <-
    function(data  ,
             time_period_length = 20 ,
             k = 5) {

        set.seed(100)
        # convert to second
        time_per_len <- lubridate::seconds(x = time_period_length)

        # Create a copy, beacuase we don't want the previously imputed data
        # have an impact on the current NA imputation process
        main_df <- data

        # obtain the NA indices
        na_index <- main_df$x_axis %>% is.na() %>%  which()


        # For each NA we need a window of data
        for (i in na_index) {
            lower_time <- main_df$record_time[i] - time_per_len
            upper_time <- main_df$record_time[i] + time_per_len
            working_df <-
                data %>%  filter(record_time < upper_time &
                                     record_time > lower_time) %>%
                na.omit()


            # Randomy sampling to allocate a value to the NA
            working_df %<>%
                data.frame %>%
                dplyr::sample_n(., k , replace = F)

            main_df$x_axis[i] <- working_df$x_axis %>%
                mean(na.rm = T)

            main_df$y_axis[i] <- working_df$y_axis %>%
                mean(na.rm = T)

            main_df$z_axis[i] <- working_df$z_axis %>%
                mean(na.rm = T)

            main_df$participant_id[i]  <-
                working_df$participant_id %>%
                mean(na.rm = T)

            other_columns <-
                working_df %>%  select(-c(record_time, x_axis, y_axis , z_axis , participant_id)) %>%
                colnames()
            # fill the nonnumerical values
            main_df %<>%  fill(other_columns)
        }


        return(main_df)
    }
