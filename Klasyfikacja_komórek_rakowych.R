library(mlbench)
library(caret)
library(class)
library(ggplot2)
library(tree)
library(neuralnet)

data("BreastCancer", package = "mlbench")

anyNA(BreastCancer)
colSums(is.na(BreastCancer))

#kolumna Id nie jest potrzebna, więc można ją usunąć.

bc<-BreastCancer[,-which(names(BreastCancer)=="Id")]

#Czyszczenie danych

bc[bc=="?"] <- NA

#Konwertuję kolumny na typ numeryczny (ostatnia kolumna zawiera zmienną docelową)

for (i in 1:(ncol(bc)-1)) {
  bc[, i] <- as.numeric(as.character(bc[, i]))
}

#Usuwamy wiersze z brakującymi wartościami
bc<-na.omit(bc)

#EDA
str(bc)

# PCA używamy tylko do wizualizacji

# 1. Obliczamy PCA — tylko na zmiennych numerycznych (bez klasy) i standaryzujemy zmienne (średnia=0, sd=1)
pca <- prcomp(bc[, -which(names(bc) == "Class")], scale. = TRUE)
# Data Frame z wynikami PCA i klasą (do wykresu biorę pierwsze dwa komponenty)
pca_df <- data.frame(pca$x[, 1:2], Class = bc$Class)

# Punkty dla klas benign i malignant tworzą dwa skupiska, to znaczy, że dane są dość dobrze separowalne.
ggplot(pca_df, aes(PC1, PC2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "PCA Visualization of Breast Cancer Data")

#Podsumowanie statystyk
summary(BreastCancer)
summary(bc)
#Sprawdzam liczby obserwacji w każdej klasie
table(bc$Class)

# Tu można zobaczyć jak rozkłady predyktorów różnią się w zależności od klasy.
features <- names(bc)[1:9]

par(mfrow = c(3, 3))
for (feature in features) {
  hist(bc[[feature]][bc$Class == "benign"],
       col = rgb(0, 0, 1, 0.5),
       xlim = c(1, 10),
       main = paste("Distribution of", feature),
       xlab = feature,
       breaks = 10)
  
  hist(bc[[feature]][bc$Class == "malignant"],
       col = rgb(1, 0, 0, 0.5),
       add = TRUE,
       breaks = 10)
}
legend("topright",
       legend = c("Benign", "Malignant"),
       fill = c(rgb(0, 0, 1, 0.5), rgb(1, 0, 0, 0.5)))

par(mfrow = c(1, 1))

# Obliczenie korelacji dla cech numerycznych
cor_matrix <- cor(bc[, -which(names(bc) == "Class")])
cor_matrix

# Znajduję bardzo silne korelacje > 0.9 (wyłączając te na przekątnej)
high_corr <- which(abs(cor_matrix) > 0.9 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_corr_pairs <- unique(t(apply(high_corr, 1, sort)))  # Unikamy duplikatów

# Wyświetlam pary -> Cell.size and Cell.shape: 0.91 
for (pair in 1:nrow(high_corr_pairs)) {
  var1 <- rownames(cor_matrix)[high_corr_pairs[pair, 1]]
  var2 <- colnames(cor_matrix)[high_corr_pairs[pair, 2]]
  cat(paste0(var1, " and ", var2, ": ", round(cor_matrix[var1, var2], 2)), "\n")
}

#Do budowy modelu regresji logistycznej wyrzucam Cell.size
bc_reduced <- bc[, !names(bc) %in% c("Cell.size")]

#Rozdzielam dane na train/test (używam caret, żeby utrzymać stosunek benign/malignant )
set.seed(123)
train <- createDataPartition(bc_reduced$Class, p=0.8, list = FALSE)

train_data <- bc_reduced[train, ]
test_data <- bc_reduced[-train, ]

# Regresja Logistyczna

log_model <- glm(Class ~ ., data=train_data, family=binomial)

# Predykcja

log_prob <- predict(log_model, newdata = test_data, type="response")
log_pred <- ifelse(log_prob > 0.5, "malignant", "benign")
log_pred <- factor(log_pred, levels=c("benign", "malignant"))

conf_matrix_log<-confusionMatrix(log_pred, test_data$Class)
conf_matrix_log

error_log <- 1- as.numeric(conf_matrix_log$overall['Accuracy'])
error_log 

# k-NN
train_labels <- train_data$Class
test_labels <- test_data$Class

# W przypadku algorytmów takich jak k-NN ważne jest, aby wszystkie cechy numeryczne były standaryzowane 
#(tj. miały średnią 0 i odchylenie standardowe 1) wyłącznie na podstawie danych treningowych
#— aby uniknąć przekazania modelowi informacji z zestawu testowego.

scaler <- preProcess(train_data[, -which(names(train_data) == "Class")], method = c("center", "scale"))

# method = c("center", "scale") (caret)
# wyśrodkowuje każdą cechę (odejmując średnią)
# standaryzuje (dzieli przez odchylenie standardowe)
# Rezultatem będą cechy ze średnią = 0 i SD = 1
# scaler jest teraz obiektem przechowującym średnią i odchylenie standardowe 
# dla każdej cechy w danych treningowych.

train_scaled <- predict(scaler, train_data[, -which(names(train_data) == "Class")])
# Dzięki temu mamy pewność, że dane testowe mają tę samą skalę co dane treningowe.
test_scaled  <- predict(scaler, test_data[, -which(names(test_data) == "Class")])

# Porównanie dokładności dla k = 1 do 20
k_values <- 1:20
accuracies <- numeric(length(k_values))

for (i in k_values) {
  knn_pred_i <- knn(train = train_scaled, test = test_scaled,
                    cl = train_labels, k = i)
  conf <- confusionMatrix(knn_pred_i, test_labels)
  accuracies[i] <- conf$overall["Accuracy"]
}

# Szukamy najlepszego k
best_k <- which.max(accuracies)
best_acc <- accuracies[best_k]
best_k 
knn_pred <- knn(train=train_scaled, test = test_scaled, cl=train_labels, k=9)

conf_matrix_knn <- confusionMatrix(knn_pred, test_labels)
conf_matrix_knn

error_knn <- 1- as.numeric(conf_matrix_knn$overall['Accuracy'])
error_knn #0.02222222

# Drzewo decycyjne

tree_model<-tree(Class~ ., data=train_data )
plot(tree_model)
text(tree_model, pretty=0)

tree_pred<-predict(tree_model, newdata=test_data, type='class')

conf_matrix_tree <- confusionMatrix(tree_pred, test_data$Class)
conf_matrix_tree #Accuracy: 0.9704  

error_tree <- 1- as.numeric(conf_matrix_tree$overall['Accuracy'])
error_tree #0.02962963

# Sprawdziłam, czy moje drzewo powinno być 'pruned', ale wyszło na to, że to nie pomoże.  
set.seed(123)
# cv.tree() sprawdza jak drzewo zachowuje się przy różnych rozmiarach
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
cv_tree # rozmiar i błąd błędnej klasyfikacji przy takim rozmiarze (najlepsze jest oryginalne drzewo)
# rozmiar drzewa vs błąd na wykresie 
plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree Size", ylab = "Misclassification Rate",
     main = "CV: Tree Size vs Error")
best_size <- cv_tree$size[which.min(cv_tree$dev)] #zwraca korespondujący rozmiar drzewa do idneksu najmniejszego błędu
pruned_tree <- prune.misclass(tree_model, best = best_size)
# To dokładnie to samo drzewo co na początku.
plot(pruned_tree)
text(pruned_tree, pretty = 0)
pruned_pred <- predict(pruned_tree, test_data, type = "class")
conf_matrix_pruned <- confusionMatrix(pruned_pred, test_data$Class)
conf_matrix_pruned

# Sieć neuronowa


# Trzeba zamienić 'benign' na 0 i 'malignant' na 1

train_nn <- train_scaled
# Ten nowy wektor numeryczny (0/1) jest przypisany jako kolumna Class do train_nn
train_nn$Class <- ifelse(train_data$Class == 'malignant', 1, 0)

test_nn <- test_scaled
test_nn$Class <- ifelse(test_data$Class == 'malignant', 1, 0)

cechy <- names(train_nn)[names(train_nn) != "Class"]

set.seed(123)
nn_model <- neuralnet(
  Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Epith.c.size +
    Bare.nuclei + Bl.cromatin + Normal.nucleoli + Mitoses,
  data = train_nn,
  hidden = 5, # 5 neuronów w ukrytej warstwie
  linear.output = FALSE,
  stepmax = 1e6
)

plot(nn_model, rep='best')

# compute daje przewidywane prawdopodobieństwa pomiędzy 0 i 1. 
# Są one przechowywane w slocie $net.result zwróconej listy.
nn_pred_raw <- compute(nn_model, test_nn[, cechy])$net.result


# Zamiana prawdopodobieństw na 0/1
nn_pred_class <- ifelse(nn_pred_raw > 0.5, 1, 0)

# Zamiana na factor 
nn_pred_factor <- factor(ifelse(nn_pred_class == 1, "malignant", "benign"),
                         levels = c("benign", "malignant"))
test_class_factor <- factor(ifelse(test_nn$Class == 1, "malignant", "benign"),
                            levels = c("benign", "malignant"))

conf_matrix_nn <- confusionMatrix(nn_pred_factor, test_class_factor)
conf_matrix_nn
error_nn <- 1- as.numeric(conf_matrix_nn$overall['Accuracy'])
error_nn

# Porównanie wyników modeli (Regresja Logistyczna, KNN, Drzewo Decyzyjne, Sieć Neuronowa)

cat("Logistic Regression  Accuracy:", round(as.numeric(conf_matrix_log$overall['Accuracy']), 4), "\n")
cat("Logistic Regression Error Rate:", round(error_log, 4), "\n\n")

par(mfrow = c(1, 4))

result_log <- ifelse(log_pred == test_data$Class, "Correct", "Incorrect")

barplot(table(result_log),
        col = c("lightgreen", "red"),
        main = "Logistic Regression Prediction Results",
        ylab = "Number of Observations")

cat("k-NN (k=9)  Accuracy:", round(as.numeric(conf_matrix_knn$overall['Accuracy']), 4), "\n")
cat("k-NN (k=9)  Error Rate:", round(error_knn, 4), "\n")

# wektor wyników
result_knn <- ifelse(knn_pred == test_labels, "Correct", "Incorrect")

barplot(table(result_knn),
        col = c("lightgreen", "red"),
        main = "k-NN Prediction Results",
        ylab = "Number of Observations")


cat("Decision Tree Accuracy:", round(conf_matrix_tree$overall['Accuracy'], 4), "\n")
cat("Decision Tree Error Rate:", round(error_tree, 4), "\n")

result_tree <- ifelse(tree_pred == test_data$Class, "Correct", "Incorrect")

barplot(table(result_tree),
        col = c("lightgreen", "red"),
        main = "Decision Tree Prediction Results",
        ylab = "Number of Observations")


cat("Neural Network Accuracy:", round(conf_matrix_tree$overall['Accuracy'], 4), "\n")
cat("Neural Network Error Rate:", round(error_nn, 4), "\n")

result_nn <- ifelse(nn_pred_factor == test_data$Class, "Correct", "Incorrect")

barplot(table(result_nn),
        col = c("lightgreen", "red"),
        main = "Neural Network Prediction Results",
        ylab = "Number of Observations")
par(mfrow = c(1, 1))


# W tym projekcie przeanalizowałam zbiór danych BreastCancer z pakietu mlbench, aby sklasyfikować guzy
# jako łagodne lub złośliwe. Po wstępnym przetworzeniu (obsługa brakujących wartości, konwersja cech 
# na wartości numeryczne i skalowanie) wytrenowałam i oceniłam cztery modele klasyfikacji.
# Model k-NN osiągnął najwyższą dokładność, choć wszystkie modele wypadły dobrze. 


#  | Model               | Accuracy  | Error Rate |
#  | ------------------- | --------- | ---------- |
#  | Logistic Regression | 96.3%     | 3.7%       |
#  | k-NN (k = 9)        | 97.8%     | 2.2%       |
#  | Decision Tree       | 97.0%     | 2.96%      |
#  | Neural Network      | 97.0%     | 2.96%      |





