# Otimização de Playlists Spotify

Projeto desenvolvido no âmbito da unidade curricular de **Otimização Computacional**, do Mestrado em Ciência de Dados do ISCTE – Instituto Universitário de Lisboa (2026/2027).

O objetivo é otimizar a ordem de uma playlist de 20 músicas, minimizando o custo das transições entre faixas consecutivas, considerando o tempo (BPM), energia, tonalidade e modo musical.

## Algoritmos implementados

- Simulated Annealing
- Tabu Search
- First Improvement Hill Climbing
- Best Improvement Hill Climbing

## Requisitos

Python 3, NumPy e Pandas.

Instalar dependências:

```bash
pip install numpy pandas
```

## Execução

O ficheiro `ProjetoOC.py` e o dataset `clean_dataset.csv` devem encontrar-se na mesma pasta.

Executar o projeto no terminal com:

```bash
python ProjetoOC.py
```

Para mais informações sobre a modelação do problema, metodologia, algoritmos e resultados experimentais, consultar o relatório do projeto.
