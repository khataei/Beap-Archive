#' Title Apply various ML models to your datasets and compare the results
#'
#' @param working_df input dataset for classification
#' @param model_names a list of models to be appied on the data
#' @param split_ratio training to test ratio
#' @param save_results_on_disk if set to true results are saved
#' @param return_plots if true returns plots confusion matrix
#' @param shrink the fraction of the data to be used for modeling. If modeling takes to long reduce this.
#' @param scale_center If true, centers and scale the data before modeling
#' @param cv_folds number of folds for cross-validation. Set to zero for not using cross-validation
#' @param RF_mtry minimum number of featuresto be used by the random forest algorithm
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
#' @importFrom data.table transpose
#' @import ggplot2
#' @export
#'
ApplyModels <-
    function(working_df,
             model_names = c("RF", "LDA", "NB", "SVM", "KNN" , "DT"),
             split_ratio = 0.66,
             scale_center = FALSE,
             cv_folds = 0,
             shrink = 1,
             save_results_on_disk = TRUE,
             return_plots = TRUE,
             RF_mtry = 2) {
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

        if (scale_center) {
            preProcValues <- preProcess(training_df, method = c("center", "scale"))
            training_df <- predict(preProcValues, training_df)
            testing_df <- predict(preProcValues, testing_df)
            message("Data is scaled and centered")

        }

        message(paste0("Data is devided into training and test set "))
        message(paste0(
            "Training set has ",
            nrow(training_df),
            " rows and ",
            ncol(training_df),
            " columns"
        ))
        message(paste0(
            "Testing set has ",
            nrow(testing_df),
            " rows and ",
            ncol(testing_df),
            " columns"
        ))


        # To store the model and all the performance metric
        results <- NULL

        # To store plots
        plts <- NULL

        # To store confusion matrix
        cf_mat <- NULL

        # To return main results for each model
        accuracies <- NULL

        # The end of preprocessing step
        toc()


        # ------------------------------------- LDA --------------------------------------
        if ("LDA" %in% model_names) {
            tic("LDA took")
            message("Starting LDA")
            model_name <- "lda"
            train_control_method <- "none"
            model_parameter <- 10

            if (cv_folds <= 0) {
                fitControl <-
                    trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
            {
                cv_folds  %<>%  ceiling()
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }
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

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "LDA"
            accuracies  %<>%  rbind(metrics)

            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


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

            if (cv_folds <= 0) {
                fitControl <-
                trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
                {
                cv_folds  %<>%  ceiling()
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }

            model_name <- "ranger"
            train_control_method <- "none"
            model_mtry <- RF_mtry
            model_splitrule <- "extratrees"
            model_min_node_size <- 10


            model_A <- train(
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

            pred <- stats::predict(model_A, newdata = testing_df)

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "RF"
                        accuracies  %<>%  rbind(metrics)


            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


            # Create a list of the model and the results to save
            results[["RF"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
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

            if (cv_folds <= 0) {
                fitControl <-
                    trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
            {
                cv_folds  %<>%  ceiling()
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }

            model_A <- train(
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

            pred <- stats::predict(model_A, newdata = testing_df)

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "NB"
                        accuracies  %<>%  rbind(metrics)

            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


            # Create a list of the model and the results to save
            results[["NB"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
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
            message("Starting KNN")
            tic("KNN took")

            if (cv_folds <= 0) {
                fitControl <-
                    trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
            {
                cv_folds  %<>%  ceiling()
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }

            model_name <- "kknn"
            train_control_method <- "none"
            model_kmax <- 3
            model_kernel <- "optimal" # Normal unweighted KNN
            model_distance <-
                1 # 1 for Manhatan , 2 for Euclidean distance


            model_A <- train(
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
                metric = "ROC"
            )


            pred <- stats::predict(model_A, newdata = testing_df)

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "KNN"
            accuracies  %<>%  rbind(metrics)

            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


            # Create a list of the model and the results to save
            results[["KNN"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
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

            if (cv_folds <= 0) {
                fitControl <-
                    trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
            {
                cv_folds  <-  ceiling(cv_folds)
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }

            model_name <- "svmPoly"
            train_control_method <- "none"
            model_degree <- 3
            model_scale <- 1
            model_C <- 0.01

            model_A <- train(
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

            pred <- stats::predict(model_A, newdata = testing_df)

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "SVM"
                        accuracies  %<>%  rbind(metrics)

            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )


            # Create a list of the model and the results to save
            results[["SVM"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
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

            if (cv_folds <= 0) {
                fitControl <-
                    trainControl(method = "none", classProbs = TRUE)
                message("Cross-validation is not being used, set cv_folds to a positive number to use cross-validation")
            } else if (cv_folds > 0)
            {
                cv_folds  %<>%  ceiling()
                fitControl <-
                    trainControl(method = "cv", number = cv_folds, classProbs = TRUE)
                message(paste0(cv_folds, " fold cross-validation is being used"))
            }

            model_A <- train(
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

            pred <- stats::predict(model_A, newdata = testing_df)

            # To calculate area AUC we need probabilies and predicted classes in a single dataframe
            pred_prob <-
                data.frame(obs =  testing_df$trimmed_activity,
                           pred = pred)
            pred <-
                stats::predict(model_A, newdata = testing_df, type = "prob")
            pred_prob <- bind_cols(pred_prob, pred)

            # Calculate different metrics
            metrics <-
                multiClassSummary(data = pred_prob,
                                  lev = levels(testing_df$trimmed_activity)) %>%
                as.data.frame()
            # Return the metric in a nicer format
            metric_names <- rownames(metrics)
            metrics  %<>% data.table::transpose()
            colnames(metrics) <- metric_names
            rownames(metrics) <- "DT"
                        accuracies  %<>%  rbind(metrics)

            # CF need a different format of prediction results so recalcuate
            pred <- stats::predict(model_A, newdata = testing_df)

            # Calculate confusion matrix
            cf_matrix <-
                confusionMatrix(
                    data = pred,
                    reference = testing_df$trimmed_activity,
                    mode = "prec_recall"
                )



            # Create a list of the model and the results to save
            results[["DT"]] <-
                list(
                    split_seed = seed,
                    model_name = model_name,
                    model = model_A,
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
            message(paste0("The models are stored in ", fname))
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
