# Mestrado em Engenharia Informática - UMinho
## Aprendizagem Automática 2
## Trabalho Prático - 2019/2020

### Contribuidores

[João Imperadeiro](https://github.com/JRI98)  
[José Boticas](https://github.com/SacitobJose)  
[Nelson Teixeira](https://github.com/Nelson198)  
[Rui Meira](https://github.com/ruisteve)

### Tema

O tema escolhido pelos 4 elementos deste grupo corresponde ao desenvolvimento de uma ***framework de AutoML***. 

Esta *framework* visa obter o melhor modelo para problemas de *supervised learning* e *unsupervised learning*, de forma automática e com a menor intervenção possível por parte do programador.

O objetivo final é colocar a *framework* disponivel para os utilizadores da linguagem *Python*.

### Planificação

Devido à complexidade do desenvolvimento deste projeto e atendendo ao curto espaço de tempo disponível, não serão incluídas opções de pré-processamento de dados. Com isto, o utilizador desta *framework* deverá indicar qual o tipo de modelo (regressão ou classificação) que deseja obter, sendo depois da responsabilidade da mesma a procura do melhor modelo desse tipo, visitando todos os algoritmos disponíveis. Se o utilizador preferir um algoritmo específico poderá indicá-lo, sendo da responsabilidade da *framework* a procura dos melhores hiperparâmetros.

A *framework* proposta procede à distinção entre problemas de *supervised* e *unsupervised learning* consoante receba, ou não, as variáveis denominadas por *labels* e *targets*. Todos os problemas de *supervised learning* vão ser distinguidos entre regressão e classificação, dependendo se a incógnita *target* é uma variável contínua ou discreta.

Para problemas de regressão, os algoritmos disponiveis são:

* Regressão Linear;
* Regressão Polinomial;
* *Support Vector Regression*;
* *Decision Tree Regression*;
* *Random Forest Regression*;
* Redes Neuronais.

Para problemas de classificação, os algoritmos disponiveis são:

* Regressão Logística;
* *k-Nearest Neighbors* (KNN);
* *Support Vector Machine* (SVM);
* *Kernel* SVM;
* *Naive Bayes*;
* *Decision Tree Classification*;
* *Random Forest*;
* Redes Neuronais.

A procura dos melhores hiperparâmetros para o problema das redes neuronais vai ser realizada com o uso da biblioteca `kerastuner`. Para os restantes algoritmos, procede-se à utilização da função *RandomizedSearchCV*, disponibilizada pela biblioteca `sklearn`.

Na eventualidade da plataforma estar terminada e ainda existir tempo para tal, será estudada a possibilidade de se incluir o pré-processamento de dados, aumentando não só a complexidade como também a flexibilidade da *framework* desenvolvida. Consequentemente, este última abordagem potencia o teste de vários modelos de tipos distintos.

### Estrutura do projeto

    ├─ Code
    │  ├─ data
    │  │  ├─ 50_Startups.csv
    │  │  └─ Social_Network_Ads.csv
    │  ├─ tests
    │  │  ├─ 01.py
    │  │  ├─ 02.py
    │  │  └─ __init__.py
    │  ├─ unicornml
    │  │  ├─ classification
    │  │  │  └─ __init__.py
    │  │  ├─ model
    │  │  │  └─ __init__.py
    │  │  ├─ regression
    │  │  │  └─ __init__.py
    │  │  └─ __init__.py
    │  ├─ options.yaml
    │  └─ setup.py
    ├─ Proposta
    │  ├─ Proposta.pdf
    │  └─ Proposta.tex
    ├─ Relatório
    │  ├─ Relatório.pdf
    │  └─ Relatório.tex
    ├─ .gitignore
    ├─ Enunciado.pdf
    ├─ LICENSE
    └─ README.md

---

### Instalação
```bash
cd Code/
python3 setup.py install --user
```

### Testes
```bash
cd Code/
python3 setup.py test
```

### Documentação
- *Supervised learning* : https://scikit-learn.org/stable/supervised_learning.html
- *API* : https://scikit-learn.org/stable/modules/classes.html
