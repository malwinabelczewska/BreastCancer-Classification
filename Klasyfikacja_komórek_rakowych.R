# -------------------------------------------------------------------------------------------------
# Podział danych na zbiór treningowy i testowy (caret, żeby utrzymać stosunek benign/malignant )
# -------------------------------------------------------------------------------------------------
# Klasyfikacja komórek rakowych - Projekt Pracownia Analizy Danych

# ======================
# W tym projekcie przeanalizowałam zbiór danych BreastCancer z pakietu mlbench, aby sklasyfikować guzy
# jako łagodne lub złośliwe. Po wstępnym przetworzeniu (obsługa brakujących wartości, konwersja cech 
# na wartości numeryczne i skalowanie) wytrenowałam i oceniłam pięć modeli klasyfikacji: 
#Regresję Logistyczną,knn, Drzewo Decyzyjne, Sieć Neuronową i Naiwny klasyfikator Bayesa.
# ======================

# ======================
# BreastCancer (mlbench)
# ======================
# Cl.thickness      – grubość warstwy komórek nabłonkowych
# Cell.size         – jednolitość rozmiaru komórek
# Cell.shape        – jednolitość kształtu komórek
# Marg.adhesion     – przyleganie komórek brzeżnych
# Epith.c.size      – rozmiar komórek nabłonkowych
# Bare.nuclei       – liczba odsłoniętych jąder komórkowych
# Bl.cromatin       – tekstura chromatyny (barwliwość jądra)
# Normal.nucleoli   – liczba zwykłych jąderek
# Mitoses           – liczba mitoz
# Class             – klasa diagnozy: „benign” (łagodny) lub „malignant” (złośliwy)


# --------------------------------------------------
# Wczytanie wymaganych bibliotek
# --------------------------------------------------
library(mlbench)
library(caret)
library(class)
library(ggplot2)
library(reshape2)
library(tree)
library(neuralnet)
library(e1071)

# --------------------------------------------------
# Wczytanie i wstępna analiza danych
# --------------------------------------------------
data("BreastCancer", package = "mlbench")

# Usunięcie zbędnej kolumny ID
bc <- BreastCancer[, -which(names(BreastCancer) == "Id")]

# Zamiana znaków '?' na NA (braki danych)
bc[bc == "?"] <- NA

# Konwersja zmiennych na typ numeryczny (ostatnia kolumna zawiera zmienną docelową)
for (i in 1:(ncol(bc) - 1)) {
  bc[, i] <- as.numeric(as.character(bc[, i]))
}

# Usunięcie obserwacji zawierających NA
bc <- na.omit(bc)

# --------------------------------------------------
# Eksploracyjna analiza danych
# --------------------------------------------------

# Struktura i statystyki opisowe
data.frame(str = str(bc))
summary(bc)

# Rozkład klas (łagodne vs złośliwe)
table(bc$Class)

# Wizualizacja PCA (tylko na zmiennych numerycznych (bez klasy) i standaryzujemy zmienne (średnia=0, sd=1))
pca <- prcomp(bc[, -which(names(bc) == "Class")], scale. = TRUE)
# Data Frame z wynikami PCA i klasą (do wykresu biorę pierwsze dwa komponenty)
pca_df <- data.frame(pca$x[, 1:2], Class = bc$Class)

ggplot(pca_df, aes(PC1, PC2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "PCA - wizualizacja danych nowotworów piersi")

# Histogramy cech z podziałem na klasy
features <- names(bc)[1:9]
par(mfrow = c(3, 3))
for (feature in features) {
  hist(bc[[feature]][bc$Class == "benign"],
       col = rgb(0, 0, 1, 0.5),
       xlim = c(1, 10),
       main = paste("Rozkład cechy:", feature),
       xlab = feature,
       breaks = 10)
  hist(bc[[feature]][bc$Class == "malignant"],
       col = rgb(1, 0, 0, 0.5),
       add = TRUE,
       breaks = 10)
}
legend("topright", legend = c("Łagodny", "Złośliwy"),
       fill = c(rgb(0, 0, 1, 0.5), rgb(1, 0, 0, 0.5)))
par(mfrow = c(1, 1))

# Macierz korelacji - Chcemy wyłączyć bardzo silne korelacje > 0.9 (wyłączając te na przekątnej)
cor_matrix <- cor(bc[, -which(names(bc) == "Class")])

# Tworzenie heatmapy

# cor_melted staje się uporządkowaną ramką danych z 3 kolumnami:
# Var1: nazwa zmiennej z wiersza
# Var2: nazwa zmiennej z kolumny
# value: korelacja między Var1 i Var2
# Ten format jest niezbędny do zbudowania mapy cieplnej ggplot2.

cor_melted <- melt(cor_matrix)
ggplot(data = cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                       limit = c(-1, 1), name = "Korelacja") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(title = "Mapa cieplna korelacji", x = "", y = "")

high_corr <- which(abs(cor_matrix) > 0.9 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_corr_pairs <- unique(t(apply(high_corr, 1, sort)))
for (pair in 1:nrow(high_corr_pairs)) {
  var1 <- rownames(cor_matrix)[high_corr_pairs[pair, 1]]
  var2 <- colnames(cor_matrix)[high_corr_pairs[pair, 2]]
  cat(paste0(var1, " i ", var2, ": korelacja = ", round(cor_matrix[var1, var2], 2)), "\n")
}

# Usunięcie cechy o wysokiej korelacji
bc_reduced <- bc[, !names(bc) %in% c("Cell.size")]

# -------------------------------------------------------------------------------------------------
# Podział danych na zbiór treningowy i testowy (caret, żeby utrzymać stosunek benign/malignant )
# -------------------------------------------------------------------------------------------------
set.seed(123)
train_index <- createDataPartition(bc_reduced$Class, p = 0.8, list = FALSE)
train_data <- bc_reduced[train_index, ]
test_data <- bc_reduced[-train_index, ]

# --------------------------------------------------
# Regresja logistyczna
# --------------------------------------------------
log_model <- glm(Class ~ ., data = train_data, family = binomial)
log_prob <- predict(log_model, newdata = test_data, type = "response")
log_pred <- ifelse(log_prob > 0.5, "malignant", "benign")
log_pred <- factor(log_pred, levels = c("benign", "malignant"))
conf_matrix_log <- confusionMatrix(log_pred, test_data$Class)
error_log <- 1 - as.numeric(conf_matrix_log$overall['Accuracy'])

# --------------------------------------------------
# Algorytm k-NN
# --------------------------------------------------
train_labels <- train_data$Class
test_labels <- test_data$Class
#-----------------------------------------------------------------------------------------
# W przypadku algorytmów takich jak k-NN ważne jest, aby wszystkie cechy
#numeryczne były standaryzowane (tj. miały średnią 0 i odchylenie standardowe 1)
#yłącznie na podstawie danych treningowych — aby uniknąć przekazania modelowi 
#informacji z zestawu testowego.
#-----------------------------------------------------------------------------------------

# Skalowanie danych
scaler <- preProcess(train_data[, -which(names(train_data) == "Class")], method = c("center", "scale"))
train_scaled <- predict(scaler, train_data[, -which(names(train_data) == "Class")])
test_scaled  <- predict(scaler, test_data[, -which(names(test_data) == "Class")])

# train_scaled: dane treningowe (bez kolumny Class), wystandaryzowane
# test_scaled: dane testowe, wystandaryzowane tą samą metodą

# method = c("center", "scale") (caret)
# wyśrodkowuje każdą cechę (odejmując średnią)
# standaryzuje (dzieli przez odchylenie standardowe)
# Rezultatem będą cechy ze średnią = 0 i SD = 1
# scaler jest teraz obiektem przechowującym średnią i odchylenie standardowe 
# dla każdej cechy w danych treningowych.

# Powtarzanie eksperymentu
set.seed(123) 
repeat_times <- 10
k_values <- 1:20
accuracies <- numeric(length(k_values))

for (k in k_values) {
  acc_list <- numeric(repeat_times)
  for (i in 1:repeat_times) {
    pred <- knn(train = train_scaled, test = test_scaled, cl = train_labels, k = k)
    acc_list[i] <- mean(pred == test_labels)
  }
  accuracies[k] <- mean(acc_list)
}

# Wybór najlepszego k
best_k <- which.max(accuracies)

# Jeśli k jest parzyste, sprawdzamy sąsiadów nieparzystych
if (best_k %% 2 == 0) {
  k1 <- best_k + 1
  k2 <- best_k - 1
  
  # Sprawdzamy, czy mieszczą się w zakresie
  k1_acc <- if (k1 <= max(k_values)) accuracies[k1] else -Inf
  k2_acc <- if (k2 >= min(k_values)) accuracies[k2] else -Inf
  
  best_k <- if (k1_acc > k2_acc) k1 else k2
}

# Końcowy model i ocena
final_pred <- knn(train = train_scaled, test = test_scaled, cl = train_labels, k = best_k)
conf_matrix_knn <- confusionMatrix(final_pred, test_labels)
error_knn <- 1 - as.numeric(conf_matrix_knn$overall['Accuracy'])

# Wyniki
cat("Najlepsze k:", best_k, "\n")
cat("Dokładność:", 1 - error_knn, "\n")
cat("Błąd klasyfikacji:", error_knn, "\n")


# --------------------------------------------------
# Drzewo decyzyjne
# --------------------------------------------------
tree_model <- tree(Class ~ ., data = train_data)
tree_pred <- predict(tree_model, newdata = test_data, type = "class")
conf_matrix_tree <- confusionMatrix(tree_pred, test_data$Class)
plot(tree_model)
text(tree_model, pretty=0)
error_tree <- 1 - as.numeric(conf_matrix_tree$overall['Accuracy'])

# Przycinanie drzewa (sprawdzenie, czy warto)
set.seed(123)
# rozmiar i błąd różnych klasyfikacji przy danym rozmiarze (najmniejszy błąd klasyfikacji wychodzi przy nie obciętym drzewie)
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
best_size <- cv_tree$size[which.min(cv_tree$dev)]
pruned_tree <- prune.misclass(tree_model, best = best_size)
pruned_pred <- predict(pruned_tree, test_data, type = "class")
conf_matrix_pruned <- confusionMatrix(pruned_pred, test_data$Class)
# To dokładnie to samo drzewo co na początku.
plot(pruned_tree)
text(pruned_tree, pretty = 0)

# --------------------------------------------------
# Sieć neuronowa
# --------------------------------------------------
# Zamiana klas na wartości 0/1
train_nn <- train_scaled
test_nn <- test_scaled
train_nn$Class <- ifelse(train_data$Class == "malignant", 1, 0)
test_nn$Class <- ifelse(test_data$Class == "malignant", 1, 0)

features_nn <- names(train_nn)[names(train_nn) != "Class"]

# 5 neuronów w warstwie ukrytej zapewnia dobre wyniki bez nadmiernego dopasowania (overfittingu).

set.seed(123)
nn_model <- neuralnet(Class ~ ., data = train_nn,
                      hidden = 5, linear.output = FALSE, stepmax = 1e6)
plot(nn_model, rep='best')
# net.result – surowe wyjścia modelu (prawdopodobieństwa), czyli liczby z przedziału (0,1)
nn_pred_raw <- compute(nn_model, test_nn[, features_nn])$net.result
nn_pred_class <- ifelse(nn_pred_raw > 0.5, 1, 0)
nn_pred_factor <- factor(ifelse(nn_pred_class == 1, "malignant", "benign"),
                         levels = c("benign", "malignant"))
test_class_factor <- factor(ifelse(test_nn$Class == 1, "malignant", "benign"),
                            levels = c("benign", "malignant"))
conf_matrix_nn <- confusionMatrix(nn_pred_factor, test_class_factor)
error_nn <- 1 - as.numeric(conf_matrix_nn$overall['Accuracy'])

# --------------------------------------------------
# Naiwny klasyfikator Bayesa
# --------------------------------------------------
model_nb <- naiveBayes(Class ~ ., data = train_data)
pred_nb <- predict(model_nb, test_data)

conf_matrix_nb <- confusionMatrix(pred_nb, test_data$Class)
error_nb <- 1 - as.numeric(conf_matrix_nb$overall['Accuracy'])


# --------------------------------------------------
# Porównanie skuteczności modeli
# --------------------------------------------------
cat("\n--- Porównanie modeli ---\n")
cat("Regresja logistyczna - trafność:", round(1 - error_log, 4), "\n")
cat("k-NN (k=", best_k, ") - trafność:", round(1 - error_knn, 4), "\n")
cat("Drzewo decyzyjne - trafność:", round(1 - error_tree, 4), "\n")
cat("Sieć neuronowa - trafność:", round(1 - error_nn, 4), "\n")
cat("Naiwny klasyfikator Bayesa - trafność:", round(1 - error_nb, 4), "\n")

# --------------------------------------------------
# Wykres porównujący trafność modeli
# --------------------------------------------------
accuracy_df <- data.frame(
  Model = c("Regresja logistyczna", paste0("k-NN (k=", best_k, ")"), "Drzewo decyzyjne", "Sieć neuronowa", "Naiwny klasyfikator Bayesa"),
  Trafnosc = c(1 - error_log, 1 - error_knn, 1 - error_tree, 1 - error_nn, 1 - error_nb)
)

ggplot(accuracy_df, aes(x = Model, y = Trafnosc)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Porównanie trafności modeli", y = "Trafność (Accuracy)", x = "Model") +
  geom_text(aes(label = round(Trafnosc, 4)), vjust = 1.5, color = "white") +
  theme_minimal()
par(mfrow = c(1, 1))
# Model k-NN osiągnął najwyższą dokładność, choć wszystkie modele wypadły dobrze. 

#  | Model               | Accuracy  | Error Rate |
#  | ------------------- | --------- | ---------- |
#  | Logistic Regression | 96.3%     | 3.7%       |
#  | k-NN (k = 9)        | 97.8%     | 2.2%       |
#  | Decision Tree       | 97.0%     | 2.96%      |
#  | Neural Network      | 97.0%     | 2.96%      |
#  | Naive Bayes         | 96.3%     | 3.7%       |
