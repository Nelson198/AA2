# Mestrado em Engenharia Informática - UMinho

## Aprendizagem Automática 2
## Trabalho Prático - 2019/2020

![Logo](https://github.com/Nelson198/AA2/blob/master/Logo.png)

### Contribuidores

[João Imperadeiro](https://github.com/JRI98)  
[José Boticas](https://github.com/SacitobJose)  
[Nelson Teixeira](https://github.com/Nelson198)  
[Rui Meira](https://github.com/ruisteve)

### Tema

O tema escolhido pelos 4 elementos deste grupo corresponde à opção relativa ao desenvolvimento de algoritmos / *software* no âmbito de aprendizagem máquina.
Mais especificamente, optou-se pelo desenvolvimento de uma ***framework de AutoML***.

Esta *framework* visa obter o melhor modelo para problemas de *supervised learning* de forma automática e com a menor intervenção possível por parte do programador.
À *framework* idealizada foi atribuído o nome *UnicornML*.

### Planificação
Dos objetivos inicialmente delineados por este grupo de trabalho, destacam-se os seguintes:
* Disponibilizar o código *open source* para todos os utilizadores de *Python* no *PyPI*;
* Servir esta *framework* como uma excelente base para um projeto de maiores dimensões;
* Implementar uma aplicação simples de utilizar, sendo apenas necessário fornecer os conjuntos de dados;
* Garantir a busca do modelo ótimo para um determinado conjunto de dados de forma rápida, eficiente e robusta.

### Implementação

Os problemas de aprendizagem supervisionada podem ser divididos em dois conjuntos: problemas de **regressão** e problemas de **classificação**.
Como tal, é importante perceber qual dos dois problemas se vai tratar, de forma a que se possa poupar tempo e recursos de computação na procura do melhor modelo.
Para isso, foi idealizada uma forma simples de identificar o tipo do problema. Caso a incógnita *target* de um determinado conjunto de dados seja uma variável discreta (valores *float*), o problema em causa é de classificação. Se esta última corresponder a uma variável contínua (valores inteiros) então trata-se de um problema de regressão.
No entanto, este processamento é evitado se o utilizador indicar qual dos problemas os seus dados representam.

A *UnicornML* oferece também diversos algoritmos para cada um dos tipos de problemas mencionados acima.
O utilizador pode escolher, dentro dos algoritmos disponíveis, quais os que quer que sejam testados. 
No entanto, estes só serão testados se estiverem disponíveis para o tipo de problema identificado pela *framework* ou indicado pelo mesmo.
Caso o utilizador não indique quais os algoritmos que prefere que sejam testados, a *framework* testará todos os algoritmos disponíveis para o tipo de problema identificado pela mesma ou indicado pelo utilizador.

Para problemas de regressão, os algoritmos disponiveis são:

* Regressão Linear;
* *Support Vector Regression* (SVR);
* Árvores de decisão;
* *Random Forest*;
* Redes Neuronais.

Para problemas de classificação, os algoritmos disponiveis são:

* Regressão Logística;
* *k-Nearest Neighbors* (KNN);
* *Support Vector Classifier* (SVC);
* *Kernel* SVM;
* *Naive Bayes* (*GaussianNB* e *BernoulliNB*);
* Árvores de decisão;
* *Random Forest*;
* Redes Neuronais.

A procura dos melhores hiperparâmetros para o problema das redes neuronais vai ser realizada com o uso da biblioteca `kerastuner`. Para os restantes algoritmos, procede-se à utilização da função *RandomizedSearchCV*, disponibilizada pela biblioteca `sklearn`.

### Estrutura do projeto

    ├─ Code
    │  ├─ data
    │  │  ├─ 50_startups.csv
    │  │  ├─ abalone.csv
    │  │  ├─ bank_notes.csv
    │  │  ├─ housing.csv
    │  │  ├─ ionosphere.csv
    │  │  ├─ iris.csv
    │  │  ├─ pregnant.csv
    │  │  ├─ social_network_ads.csv
    │  │  ├─ sonar.csv
    │  │  ├─ swedish.csv
    │  │  ├─ wheat.csv
    │  │  └─ winequality-white.csv
    │  ├─ tests
    │  │  ├─ 00.py
    │  │  ├─ 01.py
    │  │  ├─ 02.py
    │  │  ├─ 03.py
    │  │  ├─ 04.py
    │  │  ├─ 05.py
    │  │  ├─ 06.py
    │  │  ├─ 07.py
    │  │  ├─ 08.py
    │  │  ├─ 09.py
    │  │  ├─ 10.py
    │  │  ├─ 11.py
    │  │  ├─ 12.py
    │  │  └─ __init__.py
    │  ├─ unicornml
    │  │  ├─ classification
    │  │  │  └─ __init__.py
    │  │  ├─ images
    │  │  │  └─ __init__.py
    │  │  ├─ model
    │  │  │  └─ __init__.py
    │  │  ├─ neuralnetwork
    │  │  │  └─ __init__.py
    │  │  ├─ preprocessing
    │  │  │  └─ __init__.py
    │  │  ├─ regression
    │  │  │  └─ __init__.py
    │  │  └─ __init__.py
    │  ├─ options.yaml
    │  └─ setup.py
    ├─ Presentation
    │  ├─ Diagram.png
    │  └─ Presentation.pptx
    ├─ Proposal
    │  ├─ Proposal.pdf
    │  └─ Proposal.tex
    ├─ Report
    │  ├─ V1 - Initial
    │  │  ├─ Image
    │  │  │  └─ EEUM_logo.png
    │  │  ├─ Report.pdf
    │  │  └─ Report.tex
    │  ├─ V2 - Final
    │  │  ├─ Images
    │  │  │  ├─ Diagram.png
    │  │  │  └─ EEUM_logo.png
    │  │  ├─ Report.pdf
    │  │  └─ Report.tex
    ├─ .gitignore
    ├─ LICENSE
    ├─ Logo.png
    ├─ Project.pdf
    └─ README.md

Dado que este trabalho diverge de outros propostos pelos alunos desta unidade curricular, os elementos que compõem este grupo optaram pela elaboração de um relatório. Neste evidenciam-se todos os detalhes do trabalho prático, quer a nível computacional quer ao nível da estrutura da *framework* desenvolvida. Para além disso, são também exibidos
todos os testes realizados sobre os conjuntos de dados disponibilizados pela mesma. Desta forma foi possível extrair resultados pernitentes na análise da plataforma em questão, validando, consequentemente, o seu funcionamento.

O documento em causa pode ser consultado a partir do seguinte [link](https://github.com/Nelson198/AA2/blob/master/Report/V2%20-%20Final/Report.pdf) ou, de forma mais simples, pode ser vizualizado diretamente a partir da diretoria `Report/V2 - Final/Report.pdf`.

---


### Instalação
```bash
cd Code/
python3 setup.py install --user
```

### Testes
* Execução dos testes associados a todos os conjuntos de dados disponíveis:
```bash
cd Code/
python3 setup.py test
```
* Execução de um teste associado a um conjunto de dados específico:
```bash
cd Code/
python3 tests/file.py
```

### Documentação
* ***Python 3***:
  * *API* : https://docs.python.org/3/
  * ***Pandas*** : https://pandas.pydata.org/docs/
  * ***Numpy*** : https://numpy.org/doc/
  * ***Scipy*** : https://docs.scipy.org/doc/scipy/reference/
* ***sklearn***:
  * *Supervised learning* : https://scikit-learn.org/stable/supervised_learning.html
  * *API* : https://scikit-learn.org/stable/modules/classes.html
* ***tensorflow***:
  * *keras* : https://www.tensorflow.org/api_docs/python/tf/keras
* ***kerastuner***:
  * Página oficial : https://keras-team.github.io/keras-tuner/
