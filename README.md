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
- https://scikit-learn.org/stable/supervised_learning.html
- https://scikit-learn.org/stable/modules/classes.html
