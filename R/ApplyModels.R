#' Title Apply various ML models to your datasets and compare the results
#'
#' @param working_df input dataset for classification
#' @param model_names a list of models to be appied on the data
#' @param split_ratio training to test ratio
#' @param save_results if set to true results are saved
#' @param plot if true plots confusion matrix
#' @param class_column_index the column index of the classes
#'
#' @return accuracies a dataframe containing model names and accuracies
#'
#' @import caret
#' @importFrom rlang .data
#' @export
#'
ApplyModels <-
  function(working_df,
             model_names = c("RF", "LDA", "NB", "SVM", "KNN"),
             class_column_index = -1,
             split_ratio = 0.66,
             save_results = TRUE,
             plot = TRUE) {
    # Create train and test to train and evalute the model
    seed <- 2020
    set.seed(seed)
    training_indices <-
      createDataPartition(working_df$trimmed_activity,
        p = split_ratio,
        list = FALSE
      )

    training_df <- working_df %>% slice(training_indices)
    testing_df <- working_df %>% slice(-training_indices)

    # To store the model and all the performance metric
    results <- NULL

    # To return main results for each model
    accuracies <- data.frame(row.names = model_names)


    # ------------------------------------- LDA --------------------------------------
    if ("LDA" %in% model_names) {
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

      pred <- predict(model_A, newdata = testing_df)

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
        F1_Score(
          y_true = testing_df$trimmed_activity,
          y_pred = pred
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

      if (plot) {
        cf_matrix$table %>%
          data.frame() %>%
          ggplot(aes(Prediction, Reference)) +
          geom_tile(aes(fill = Freq), colour = "gray50") +
          scale_fill_gradient(low = "beige", high = muted("chocolate")) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
          ggtitle(model_name)
      }
    }


    #------------------------------ Random Forest _ Ranger package----------------------
    if ("RF" %in% model_names) {
      # Ranger is a fast implementation of random forests (Breiman 2001)
      # The method is none becuase we have test and train data
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

      pred <- predict(model_A_rf, newdata = testing_df)

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
        F1_Score(
          y_true = testing_df$trimmed_activity,
          y_pred = pred
        )

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


      if (plot) {
        cf_matrix$table %>%
          data.frame() %>%
          ggplot(aes(Prediction, Reference)) +
          geom_tile(aes(fill = Freq), colour = "gray50") +
          scale_fill_gradient(low = "beige", high = muted("chocolate")) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
          ggtitle(model_name)
      }
    }



    #---------------------------------- Naive Bayes Classifier ----------------------------
    if ("NB" %in% model_names) {
      # The method is none becuase we have test and train data
      model_name <- "nb"
      train_control_method <- "none"
      model_fL <- 3
      model_usekernel <- TRUE
      model_adjust <- 1.5

      fitControl <-
        trainControl(method = train_control_method, classProbs = TRUE)


      model_A_nb <- train(
        trimmed_activity ~ .,
        data = training_df,
        method = model_name,
        trControl = fitControl,
        verbose = FALSE,
        tuneGrid = data.frame(
          fL = model_fL,
          usekernel = model_usekernel,
          adjust = model_adjust
        ),
        metric = "ROC"
      )

      pred <- predict(model_A_nb, newdata = testing_df)

      cf_matrix <-
        confusionMatrix(
          data = pred,
          reference = testing_df$trimmed_activity,
          mode = "prec_recall"
        )
      # Calculate accuracy and F1
      accuracies["NB", "Acc"] <-
        mean(pred == testing_df$trimmed_activity)

      accuracies["NB", "F1"] <-
        F1_Score(
          y_true = testing_df$trimmed_activity,
          y_pred = pred
        )

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
      if (plot) {
        cf_matrix$table %>%
          data.frame() %>%
          ggplot(aes(Prediction, Reference)) +
          geom_tile(aes(fill = Freq), colour = "gray50") +
          scale_fill_gradient(low = "beige", high = muted("chocolate")) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
          ggtitle(model_name)
      }
    }




    #----------------------------------- k-Nearest Neighbors -----------------------------------
    if ("KNN" %in% model_names) {
      # using kknn package
      # The method is none becuase we have test and train data
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

      pred <- predict(model_A_kknn, newdata = testing_df)

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
        F1_Score(
          y_true = testing_df$trimmed_activity,
          y_pred = pred
        )

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

      if (plot) {
        cf_matrix$table %>%
          data.frame() %>%
          ggplot(aes(Prediction, Reference)) +
          geom_tile(aes(fill = Freq), colour = "gray50") +
          scale_fill_gradient(low = "gray99", high = muted("deepskyblue4")) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      }
    }

    #------------------------------ Support Vector Machines with Polynomial Kernel ----------------------
    if ("SVM" %in% model_names) {
      # using  kernlab package
      # The method is none becuase we have test and train data
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

      pred <- predict(model_A_svm, newdata = testing_df)

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
        F1_Score(
          y_true = testing_df$trimmed_activity,
          y_pred = pred
        )

      # Create a list of the model and the results to save
      results[["SVM"]] <-
        list(
          split_seed = seed,
          model_name = model_name,
          model = model_A_svm,
          train_control_method = train_control_method,
          tune_parameters = c(model_degree, model_scale, model_C),
          cf_matrix = cf_matrix,
          predictions = pred
        )
    }


    # ---------------------------- save the results ----------------------

    if (save_results) {
      save(results,
        file = paste0("Model_results_", as.numeric(now()), ".RData")
      )
    }
    return(accuracies)
  }
