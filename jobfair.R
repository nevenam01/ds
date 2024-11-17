#install.packages(c("dplyr", "tidyr", "ggplot2", "caret", "randomForest", "forcats", "doParallel"))
#ucitavanje potrebnih paketa
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(randomForest)
library(forcats)
library(doParallel)

registration_data <- read.csv("registration_data_training.csv")
registration_data_test <- read.csv("registration_data_test.csv")
previous_lives_training <- read.csv("previous_lives_training_data.csv")
previous_lives_test <- read.csv("previous_lives_test_data.csv")

str(registration_data)
str(previous_lives_training)

#spajanje trening i test tabela
combined_training_data <- left_join(registration_data, previous_lives_training, by = "user_id")
combined_test_data <- left_join(registration_data_test, previous_lives_test, by = "user_id")

str(combined_training_data)
str(combined_test_data)

#previse razlicitih vrednosti
length(unique(combined_training_data$registration_device_manufacturer))
combined_training_data$registration_device_manufacturer <- NULL
combined_training_data$registration_country.x <- NULL
combined_training_data$registration_country.y <- NULL
combined_training_data$registration_time_utc <- NULL
combined_training_data$registration_date <- NULL
combined_training_data <- combined_training_data %>% select(-ends_with(".y"))
combined_training_data$registration_store.x <- NULL

combined_test_data$registration_device_manufacturer <- NULL
combined_test_data$registration_country.x <- NULL
combined_test_data$registration_country.y <- NULL
combined_test_data$registration_time_utc <- NULL
combined_test_data$registration_date <- NULL
combined_test_data <- combined_test_data %>% select(-ends_with(".y"))
combined_test_data$registration_store.x <- NULL

apply(combined_training_data,2,function(x) sum(is.na(x))) #avg_age_top_11_players 5 na vrednosti
apply(combined_training_data,2,function(x) sum(x=="",na.rm = T))
apply(combined_training_data,2,function(x) sum(x==" ",na.rm = T))
apply(combined_training_data,2,function(x) sum(x=="-",na.rm = T))

combined_training_data$avg_age_top_11_players[is.na(combined_training_data$avg_age_top_11_players)]<-median(combined_training_data$avg_age_top_11_players,na.rm=T)

#pretvori kategorijske kolone u faktore
categorical_cols <- c("registration_platform_specific.x", "registration_channel_detailed.x",
                      "registration_device_type", "is_payer_lifetime",
                      "is_rewarded_video_watcher_lifetime")

prop.table(table(combined_training_data$registration_channel_detailed.x))
# Pretvori u faktore
combined_training_data[categorical_cols] <- lapply(combined_training_data[categorical_cols], as.factor)
combined_test_data[categorical_cols] <- lapply(combined_test_data[categorical_cols], as.factor)

summary(combined_training_data)
summary(combined_test_data)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#agregacija podataka po user_id
combined_training_data <- combined_training_data %>%
  group_by(user_id) %>%
  summarise(
    #prosek za numericke kolone
    across(where(is.numeric), mean, na.rm = TRUE),
    
    #prosek za kategorijske kolone
    across(where(is.factor), ~ Mode(.))
  )

str(combined_training_data)

#agregacija podataka po user_id
combined_test_data <- combined_test_data %>%
  group_by(user_id) %>%
  summarise(
    #prosek za numericke kolone
    across(where(is.numeric), mean, na.rm = TRUE),
    
    #prosek za kategorijske kolone
    across(where(is.factor), ~ Mode(.))
  )

str(combined_test_data)

combined_training_data$user_id <- NULL
combined_test_data$user_id <- NULL

#random forest model
# Ispravljena mreža hiperparametara
tuneGrid <- expand.grid(
  mtry = c(5, 10)  # Broj promenljivih koje model razmatra po čvoru
)

# Postavi paralelno procesiranje sa brojem jezgara
cl <- makeCluster(4)  # Broj jezgara prilagodi svom računaru
registerDoParallel(cl)

# Kontrola treninga sa unakrsnom validacijom
train_control <- trainControl(
  method = "cv",    # Unakrsna validacija
  number = 3,       # 3 folda
  allowParallel = TRUE  # Omogući paralelno procesiranje
)

# Treniranje Random Forest modela
set.seed(123)
rf_model <- train(
  days_active_first_28_days_after_registration ~ ., 
  data = combined_training_data, 
  method = "rf", 
  trControl = train_control, 
  tuneGrid = tuneGrid,
  ntree = 300,          # Broj stabala
  nodesize = 5          # Minimalna veličina čvorova
)

# Zaustavi paralelno procesiranje
stopCluster(cl)
registerDoSEQ()

# Prikaži najbolje hiperparametre
print(rf_model$bestTune)

# Prikaz važnosti promenljivih
varImpPlot(rf_model$finalModel)


#predikcije na validacionom skupu
validation_preds <- predict(rf_model, combined_test_data)

#Mean Absolute Error
mae <- mean(abs(validation_preds - combined_test_data$days_active_first_28_days_after_registration))
print(paste("MAE:", mae))

# Predikcije na test podacima
test_preds <- predict(rf_model, combined_test_data)

#cuvanje rezultata
submission <- data.frame(
  user_id = combined_test_data$user_id[1:length(test_preds)],
  predicted_days_active_first_28_days_after_registration = test_preds
)

write.csv(submission, "days_active_first_28_days_after_registration_predictions.csv", row.names = FALSE)

# Sačuvaj trening podatke u CSV fajl
write.csv(combined_training_data, "combined_training_data.csv", row.names = FALSE)



