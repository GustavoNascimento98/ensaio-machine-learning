# Ensaio de Machine Learning
![Machine Learning](img/project_picture.jpg)

## Problema de negócio
- **Descrição**    
    A empresa Data Money acredita que a expertise no treinamento e ajuste fino dos algoritmos, feito pelos Cientistas de Dados da empresa, é o principal motivo dos ótimos resultados que as consultorias vem entregando aos seus clientes.
        
- **Objetivo**
    O objetivo desse projeto será realizar ensaios com algoritmos de Classificação, Regressão e Clusterização, para estudar a mudança de comportamento da performance, medida que os valores dos parâmetros de controle de overfitting e underfitting mudam.
        
    

## Planejamento da solução
- Produto final
    - O produto final será 7 tabelas mostrando a performance dos algoritmos, avaliados usando múltiplas métricas, para 3 conjuntos de dados diferentes: Treinamento, Validação e Teste.
        
- Algoritmos ensaiados
        
    - **Classificação**
        
        ---    
        Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression.
        
        Métricas de performance: Accuracy, Precision, Recall e F1-Score.
        
    - **Regressão**
        
        ---
        
        Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polynomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regresion Elastic Net, Polynomial Regression Lasso, Polynomial Regression Ridge e Polynomial Regression Elastic Net.
        
        Métricas de performance: R2, MSE, RMSE, MAE e MAPE.
        
        **************************Clusterização**************************
        
        ---
        
        Algoritmos: K-Means e Affinity Propagation.
        
        Métricas de performance: Silhouette Score.
        
- Ferramentas utilizadas
        
    - Python 3.10 e Scikit-learn
        

## Desenvolvimento

**Estratégia da solução**
        
Para o objetivo de ensaiar os algoritmos de Machine Learning, eu vou escrever os códigos utilizando a linguagem Python, para treinar cada um dos algoritmos e vou variar seus principais parâmetros de ajuste de overfitting e observar a métrica final.
        
O conjunto de valores que fizerem os algoritmos alcançarem a melhor performance, serão aqueles escolhidos para o treinamento final do algoritmo.
        

<details>
<summary><b> Passo a passo </b></summary>    

1. Divisão dos dados em treino, teste e validação.

2. Treinamento dos algoritmos com os dados de treinamento, usando os parâmetros “default”.

3. Medir a performance dos algoritmos treinados com os parâmetros “default”, usando o próprio conjunto de dados de treinamento.

4. Medir a performance dos algoritmos treinados com os parâmetros “default”, usando o conjunto de dados de validação.

5. Alternar os valores dos principais parâmetros que controlam o overfitting do algoritmo até encontrar o conjunto de parâmetros que apresente a melhor performance dos algoritmos.

6. Unir os dados de treinamento e validação.

7. Retreinar o algoritmo com a união dos dados de treinamento e validação, utilizando os melhores valores para os parâmetros de controle do algoritmo.

8. Medir a performance dos algoritmos treinados com os melhores parâmetros, utilizando o conjunto de dados de teste.

9. Avaliar os ensaios e anotar os 3 principais insights que se destacaram.
</details>
        


## Os top 3 insights

 **Insight top 1**
        
Os algoritmos baseados em árvores possuem uma performance melhor em todas as métricas, quando aplicados sobre os dados de teste, no ensaio de classificação.
        
 **Insight top 2**
        
A performance dos algoritmos de classificação sobre os dados de validação ficou bem próxima da performance sobre os dados de teste.
        
 **Insight top 3**
        
Todos os algoritmos de regressão não apresentam boas métricas de performance, o que mostra uma necessidade de uma seleção de atributos e uma preparação melhor das variáveis dependentes do conjunto de dados.
        

## 6. Resultados
    
**Ensaio de classificação:**

<details>
<summary>Sobre os dados de treinamento</summary> 

| Algorithm | Accuracy | Precision | Recall | F1_score |
| --- | --- | --- | --- | --- |
| 0 | KNN | 0.934055 | 0.964572 | 0.880171 |
| 1 | Decision Tree | 0.973674 | 0.981817 | 0.956981 |
| 2 | Random Forest | 1.000000 | 1.000000 | 1.000000 |
| 3 | Logistic Regression | 0.875115 | 0.870152 | 0.836706 |
</details>

</br>

<details>
<summary>Sobre os dados de validação</summary>   
        
| Algorithm | Accuracy | Precision | Recall | F1_score |
| --- | --- | --- | --- | --- |
| 0 | KNN | 0.926510 | 0.957389 | 0.869107 |
| 1 | Decision Tree | 0.950706 | 0.954051 | 0.931101 |
| 2 | Random Forest | 0.962193 | 0.971468 | 0.940382 |
| 3 | Logistic Regression | 0.874095 | 0.869015 | 0.835400 |
</details>        

</br>

<details>
<summary>Sobre os dados de teste</summary>
       
| Algorithm | Accuracy | Precision | Recall | F1_score |
| --- | --- | --- | --- | --- |
| 0 | KNN | 0.924999 | 0.955173 | 0.869952 |
| 1 | Decision Tree | 0.951570 | 0.955574 | 0.933040 |
| 2 | Random Forest | 0.961766 | 0.970351 | 0.941663 |
| 3 | Logistic Regression | 0.871703 | 0.868573 | 0.833876 |
</details>
</br>

---

**Ensaio de regressão:**
    
<details>
<summary>Sobre os dados de treinamento</summary>         
        
| Algorithm | R2 | MSE | RMSE | MAE | MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline | 0.000000 | 478.012560 | 21.863498 | 17.365090 |
| 1 | Linear Regression | 0.046058 | 455.996112 | 21.354065 | 16.998249 |
| 2 | Decision Tree | 0.113523 | 423.747268 | 20.585122 | 16.368766 |
| 3 | Random Forest | 0.905269 | 45.282648 | 6.729238 | 4.819884 |
| 4 | Polynomial Regression | 0.094195 | 432.986210 | 20.808321 | 16.458032 |
| 5 | Linear Regression Lasso | 0.041219 | 458.309397 | 21.408162 | 17.046776 |
| 6 | Linear Regression Ridge | 0.046018 | 456.015424 | 21.354518 | 16.999016 |
| 7 | Linear Regression Elastic Net | 0.041219 | 458.309397 | 21.408162 | 17.046776 |
| 8 | Polynomial Regression Lasso | 0.067909 | 445.551320 | 21.108087 | 16.743258 |
| 9 | Polynomial Regression Ridge | 0.092837 | 433.635258 | 20.823911 | 16.476004 |
| 10 | Polynomial Regression Elastic Net | 0.067909 | 445.551320 | 21.108087 | 16.743258 |
</details>
</br>

<details>
<summary>Sobre os dados de validação</summary> 

| Algorithm | R2 | MSE | RMSE | MAE | MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline | -7.197077e-07 | 477.511956 | 21.852047 | 17.352836 |
| 1 | Linear Regression | 3.992483e-02 | 458.447042 | 21.411376 | 17.039754 |
| 2 | Decision Tree | 6.355928e-02 | 447.161319 | 21.146189 | 16.843452 |
| 3 | Random Forest | 3.371453e-01 | 316.520801 | 17.791031 | 13.008747 |
| 4 | Polynomial Regression | 6.647668e-02 | 445.768223 | 21.113224 | 16.749939 |
| 5 | Linear Regression Lasso | 3.719533e-02 | 459.750411 | 21.441791 | 17.047448 |
| 6 | Linear Regression Ridge | 3.993790e-02 | 458.440800 | 21.411231 | 17.037793 |
| 7 | Linear Regression Elastic Net | 3.719533e-02 | 459.750411 | 21.441791 | 17.047448 |
| 8 | Polynomial Regression Lasso | 5.889836e-02 | 449.386960 | 21.198749 | 16.818893 |
| 9 | Polynomial Regression Ridge | 6.770020e-02 | 445.183981 | 21.099383 | 16.739326 |
| 10 | Polynomial Regression Elastic Net | 5.889836e-02 | 449.386960 | 21.198749 | 16.818893 |
</details>
</br>

<details>
<summary>Sobre os dados de teste</summary> 
        
| Algorithm | R2 | MSE | RMSE | MAE | MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline | -0.000124 | 486.961469 | 22.067203 | 17.551492 |
| 1 | Linear Regression | 0.052317 | 461.427719 | 21.480869 | 17.129965 |
| 2 | Decision Tree | 0.072181 | 451.755789 | 21.254547 | 17.010757 |
| 3 | Random Forest | 0.356790 | 313.179616 | 17.696882 | 12.981922 |
| 4 | Polynomial Regression | 0.090079 | 443.041256 | 21.048545 | 16.720535 |
| 5 | Linear Regression Lasso | 0.044728 | 465.122726 | 21.566704 | 17.175600 |
| 6 | Linear Regression Ridge | 0.052199 | 461.485049 | 21.482203 | 17.128327 |
| 7 | Linear Regression Elastic Net | 0.044728 | 465.122726 | 21.566704 | 17.175600 |
| 8 | Polynomial Regression Lasso | 0.070407 | 452.619970 | 21.274867 | 16.912310 |
| 9 | Polynomial Regression Ridge | 0.088758 | 443.684800 | 21.063827 | 16.732546 |
| 10 | Polynomial Regression Elastic Net | 0.070407 | 452.619970 | 21.274867 | 16.912310 |
</details>
</br>

---

    
**Ensaio de clusterização:**
    
<details>
<summary>Sobre os dados de treinamento</summary>

| Algorithm | Clusters | Silhoutte Score |
| --- | --- | --- |
| 0 | KMeans | 3 |
| 1 | AffinityPropagation | 3 |
</details>
</br>   

## Conclusões
    
Nesse ensaio de Machine Learning, consegui adquirir experiência e entender melhor sobre os limites dos algoritmos entre os estados de underfitting e overfitting.

Algoritmos baseados em árvores são sensíveis quanto a profundidade do crescimento e do número de árvores na floresta, fazendo com que a escolha correta dos valores desses parâmetros impeçam os algoritmos de entrar no estado de overfitting.

Os algoritmos de regressão, por outro lado, são sensíveis ao grau do polinômio. Esse parâmetro controla o limite entre o estado de underfitting e overfitting desses algoritmos.

Esse ensaio de Machine Learning foi muito importante para aprofundar o entendimento sobre o funcionamento de diversos algoritmos de classificação, regressão e clusterização, e quais os principais parâmetros de controle entre os estados de underfitting e overfitting.


## Próximos passos
Como próximos passos desse ensaio, pretendo ensaiar novos algoritmos de Machine Learning e usar diferentes conjuntos de dados para aumentar o conhecimenyo sobre os algoritmos e quais cenários são mais favoráveis para o aumento da performance dos mesmos.