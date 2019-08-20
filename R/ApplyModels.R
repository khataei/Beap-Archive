#' Title Apply various ML models to your datasets and compare the results
#'
#' @param working_df input dataset for classification
#' @param model_names a list of models to be appied on the data
#' @param split_ratio training to test ratio
#' @param save_results_on_disk if set to true results are saved
#' @param return_plots if true returns plots confusion matrix
#' @param shrink the fraction of the data to be used for modeling. If modeling takes to long reduce this.
#'
#' @return output a list containing a dataframe containing model names and accuracies and a list of plots
#'
#' @import caret
#' @importFrom rlang .data
#' @importFrom dplyr slice
#' @importFrom tictoc tic
#' @importFrom tictoc toc
#' @importFrom parallel makePSOCKcluster
#' @importFrom doParallel registerDoParallel
#' @importFrom parallel stopCluster
#' @importFrom MLmetrics F1_Score
#' @import ggplot2
#' @export
#'
ApplyModels <-
    function(working_df,
             model_names = c("RF", "LDA", "NB", "SVM", "KNN" , "DT"),
             split_ratio = 0.66,
             shrink = 1,
             save_results_on_disk = TRUE,
             return_plots = TRUE) {
        # # Parallel and time to see if caret parallel works
        tic("Preprocessing")
        cl <- makePSOCKcluster(5)
        registerDoParallel(cl)

        # Create train and test to train and evalute the model
        seed <- 2020
        set.seed(seed)

        working_df %<>% dplyr::sample_frac(shrink)
        message(paste0(shrink * 100, " % of the data will be used"))

        training_indices <-
            createDataPartition(working_df$trimmed_activity,
                                p = split_ratio,
                                list = FALSE)

        training_df <- working_df %>% dplyr::slice(training_indices)
        testing_df <- working_df %>% dplyr::slice(-training_indices)
        message(paste0("Data is devided into training and test set "))


        # To store the model and all the performance metric
        results <- NULL

        # To store plots
        plts <- NULL

        # To store confusion matrix
        cf_mat <- NULL

        # To return main results for each model
        accuracies <- data.frame(row.names = model_names)

        # The end of preprocessing step
        toc()


        # ------------------------------------- LDA --------------------------------------
        if ("LDA" %in% model_names) {
            tic("LDA took")
            message("Starting LDA")
            model_name <- "lda"
            train_control_method <- "none"
            model_parameter <- 10

            fitControl <-
                trainControl(method = train_control_method, classProbs = TRUE)

            model_A <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(parameter = model_parameter),
                metric = "ROC"
            )

            pred <- stats::predict(model_A, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )

            # Calculate accuracy and F1
            accuracies["LDA", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["LDA", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["LDA"] <- cf_matrix

            # create a list of the model and the results to save
            results[["LDA"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_parameter),
                    cf_matrix = cf_matrix
                )

            if (return_plots) {
                plts[["LDA"]] <-  cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("LDA")
            }
            toc()
        }



        #------------------------------ Random Forest _ Ranger package----------------------
        if ("RF" %in% model_names) {
            # Ranger is a fast implementation of random forests (Breiman 2001)
            # The method is none becuase we have test and train data
            tic("RF took")
            message("Starting RF")
            fitControl <-
                trainControl(method = "none", classProbs = TRUE)
            model_name <- "ranger"
            train_control_method <- "none"
            model_mtry <- 2
            model_splitrule <- "extratrees"
            model_min_node_size <- 10


            model_A_rf <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(
                    mtry = model_mtry,
                    splitrule = model_splitrule,
                    min.node.size = model_min_node_size
                ),
                metric = "ROC"
            )

            pred <- stats::predict(model_A_rf, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


            # Calculate accuracy and F1
            accuracies["RF", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["RF", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["RF"] <- cf_matrix

            # Create a list of the model and the results to save
            results[["RF"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A_rf,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_mtry, model_splitrule, model_min_node_size),
                    cf_matrix = cf_matrix
                )


            if (return_plots) {
                plts[["RF"]] <- cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("RF")
            }
            toc()
        }



        #---------------------------------- Naive Bayes Classifier ----------------------------
        if ("NB" %in% model_names) {
            # The method is none becuase we have test and train data
            tic("NB took:")
            message("Starting NB")
            model_name <- "nb"
            train_control_method <- "none"
            model_fL <- 3
            model_usekernel <- TRUE
            model_adjust <- 1.5

            fitControl <-
                trainControl(method = train_control_method  , classProbs =  TRUE)


            model_A_nb <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(
                    fL = model_fL ,
                    usekernel = model_usekernel ,
                    adjust = model_adjust
                ),
                metric = "ROC"
            )

            pred <- stats::predict(model_A_nb, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred ,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )

            # Calculate accuracy and F1
            accuracies["NB", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["NB", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["NB"] <- cf_matrix

            # Create a list of the model and the results to save
            results[["NB"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A_nb,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_fL, model_usekernel, model_adjust),
                    cf_matrix = cf_matrix
                )
            if (return_plots) {
                plts[["NB"]] <-  cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("NB")
            }
            toc()
        }




        #----------------------------------- k-Nearest Neighbors -----------------------------------
        if ("KNN" %in% model_names) {
            # using kknn package
            # The method is none becuase we have test and train data
            tic("KNN took")
            fitControl <-
                trainControl(method = "none", classProbs = TRUE)
            model_name <- "kknn"
            train_control_method <- "none"
            model_kmax <- 10
            model_kernel <- "optimal" # Normal unweighted KNN
            model_distance <-
                2 # 1 for Manhatan , 2 for Euclidean distance


            model_A_kknn <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(
                    kmax = model_kmax,
                    kernel = model_kernel,
                    distance = model_distance
                ),
                metric = "Accuracy"
            )

            pred <- stats::predict(model_A_kknn, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )
            # Calculate accuracy and F1
            accuracies["KNN", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["KNN", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["KNN"] <- cf_matrix

            # Create a list of the model and the results to save
            results[["KNN"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A_kknn,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_kmax, model_kernel, model_distance),
                    cf_matrix = cf_matrix
                )

            if (return_plots) {
                plts[["KNN"]] <- cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("KNN")
            }
            toc()
        }

        #------------------------------ Support Vector Machines with Polynomial Kernel ----------------------
        if ("SVM" %in% model_names) {
            # using  kernlab package
            # The method is none becuase we have test and train data
            tic("SVM took")
            message("Starting SVM")
            fitControl <-
                trainControl(method = "none", classProbs = TRUE)
            model_name <- "svmPoly"
            train_control_method <- "none"
            model_degree <- 3
            model_scale <- 1
            model_C <- 0.1

            model_A_svm <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(
                    degree = model_degree,
                    scale = model_scale,
                    C = model_C
                ),
                metric = "ROC"
            )

            pred <- stats::predict(model_A_svm, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )

            # Calculate accuracy and F1
            accuracies["SVM", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["SVM", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["SVM"] <- cf_matrix

            # Create a list of the model and the results to save
            results[["SVM"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A_svm,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_degree, model_scale, model_C),
                    cf_matrix = cf_matrix
                )

            if (return_plots) {
                plts[["SVM"]] <- cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("SVM")
            }
            toc()
        }

        # -------------------------- DT ------------------------
        if ("DT" %in% model_names) {
            tic("DT took")
            message("Starting DT")
            model_name <- "C5.0"
            train_control_method <- "none"
            model_trails <- 10
            model_model <- "C5.0"
            model_winnow <- FALSE

            fitControl <-
                trainControl(method = train_control_method, classProbs = TRUE)


            model_A_DT <- train(
                trimmed_activity ~ .,
                data = training_df,
                method = model_name,
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = data.frame(
                    trials = model_trails,
                    model = model_model,
                    winnow = model_winnow
                ),
                metric = "ROC"
            )

            pred <- stats::predict(model_A_DT, newdata = testing_df)

            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )

            # Calculate accuracy and F1 and save cf_matrix
            accuracies["DT", "Acc"] <-
                mean(pred == testing_df$trimmed_activity)

            accuracies["DT", "F1"] <-
                MLmetrics::F1_Score(y_true = testing_df$trimmed_activity,
                         y_pred = pred)
            cf_mat["DT"] <- cf_matrix

            # Create a list of the model and the results to save
            results[["DT"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A_DT,
                    train_control_method = train_control_method,
                    tune_parameters = c(model_trails, model_model, model_winnow),
                    cf_matrix = cf_matrix
                )

            if (return_plots) {
                plts[["DT"]] <- cf_matrix$table %>%
                    data.frame() %>%
                    ggplot2::ggplot(aes(Prediction, Reference)) +
                    geom_tile(aes(fill = Freq), colour = "gray50") +
                    scale_fill_gradient(low = "beige", high = muted("chocolate")) +
                    geom_text(aes(label = Freq)) +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    ggtitle("DT")
            }
            toc()
        }


        # ---------------------------- save the results ----------------------

        if (save_results_on_disk) {
            fname <- paste0("Model_results_", as.numeric(now()), ".RData")
            save(results,
                 file = fname)
            message(paste0("The models are stored in", fname))
        }


        # To stop parallel calculation
        stopCluster(cl)

        output <-
            list(
                "Model-Accuracy" = accuracies,
                "Plots" = plts,
                "Confusion-Matrices" = cf_mat
            )
        return(output)
    }
