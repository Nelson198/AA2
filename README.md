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
    │  ├─ Presentation.md
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

Dado que este trabalho diverge de outros propostos pelos alunos desta unidade curricular, os elementos que compõem este grupo optaram pela elaboração de um relatório. Neste evidenciam-se todos os detalhes do trabalho prático, quer a nível computacional quer ao nível da estrutura da *framework* desenvolvida.

O documento em causa pode ser consultado a partir do seguinte [link](https://github.com/Nelson198/AA2/blob/master/Report/V2%20-%20Final/Report.pdf) ou, de forma mais simples, pode ser vizualizado diretamente a partir da diretoria `Report/V2 - Final/Report.pdf`.

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
