import math
import time
import numpy as np
import pandas as pd

# ==============================================================================
# A. CARREGAMENTO DO DATASET E DEFINIÇÃO DA AMOSTRA
# ==============================================================================
# O dataset já foi previamente limpo. A amostra é fixa e reprodutível:
# random_state=42 garante que são sempre selecionadas as mesmas 20 músicas.
df = pd.read_csv("clean_dataset.csv")

N = 20
fix = 42
playlist = df.sample(n=N, random_state=fix).reset_index(drop=True)


# ==============================================================================
# B. CONSTRUÇÃO DA MATRIZ DE CUSTOS E FUNÇÃO OBJETIVO
# ==============================================================================
# Os pesos representam a importância relativa de cada atributo na avaliação
# da descontinuidade entre duas músicas consecutivas.
w1, w2, w3, w4 = 0.45, 0.25, 0.15, 0.15

# Conversão dos atributos em vetores coluna para permitir o cálculo matricial das diferenças entre todos os pares de músicas.
bpm = playlist["tempo"].to_numpy()[:, None]
energy = playlist["energy"].to_numpy()[:, None]
key = playlist["key"].to_numpy()[:, None]
mode = playlist["mode"].to_numpy()[:, None]

# Normalização das diferenças:
# - BPM: diferença absoluta dividida por 20 e limitada ao máximo de 1;
# - Energia: diferença absoluta (os valores já se encontram entre 0 e 1);
# - Tonalidade: distância circular mínima entre as 12 tonalidades, dividida por 6;
# - Modo: diferença absoluta entre os valores binários 0 e 1.
n_bpm = np.minimum(1.0, np.abs(bpm - bpm.T) / 20.0)
n_energy = np.abs(energy - energy.T)
n_key = np.minimum(np.abs(key - key.T), 12 - np.abs(key - key.T)) / 6.0
n_mode = np.abs(mode - mode.T)

# Matriz C: custo ponderado de transição entre cada par de músicas.
# A diagonal é colocada a zero, pois não existe transição de uma música para ela própria.
C = w1 * n_bpm + w2 * n_energy + w3 * n_key + w4 * n_mode
np.fill_diagonal(C, 0.0)

# Contador global das avaliações da função objetivo.
# É reiniciado antes de cada execução experimental.
avaliacoes = 0


def custo_ordem(ordem):
    """Calcula o custo total de uma sequência linear de músicas."""
    global avaliacoes
    avaliacoes += 1

    # Soma os custos das transições entre músicas consecutivas.
    # Não existe ligação entre a última música e a primeira.
    custo_total = 0.0
    for k in range(len(ordem) - 1):
        custo_total += C[ordem[k], ordem[k + 1]]

    return custo_total


# ==============================================================================
# C. IMPLEMENTAÇÃO DOS ALGORITMOS
# ==============================================================================
# São disponibilizadas duas versões de cada algoritmo:
# - versão limitada pelo número máximo de iterações (Experiência 1);
# - versão limitada pelo número máximo de avaliações da função objetivo
#   (Experiência 2).


# ------------------------------------------------------------------------------
# C.1. SIMULATED ANNEALING — LIMITE DE ITERAÇÕES
# ------------------------------------------------------------------------------
def simulated_annealing_iterations(objective, init_order, max_iterations,
                                   temp=1.0, seed=42):
    rng = np.random.default_rng(seed)

    # A melhor solução global é mantida separadamente da solução corrente.
    best = init_order.copy()
    best_eval = objective(best)
    current = best.copy()
    current_eval = best_eval
    scores = [best_eval]

    t0 = time.perf_counter()

    for step in range(max_iterations):
        # Arrefecimento: a temperatura diminui à medida que as iterações avançam.
        t = temp / float(step + 1)

        # Geração de um vizinho através da troca aleatória de duas posições.
        candidato = current.copy()
        i, j = rng.choice(len(current), size=2, replace=False)
        candidato[i], candidato[j] = candidato[j], candidato[i]

        candidato_eval = objective(candidato)

        # Aceita sempre uma melhoria. Uma solução pior pode ser aceite
        # probabilisticamente, permitindo escapar de mínimos locais.
        if candidato_eval < current_eval or rng.random() < math.exp(
            (current_eval - candidato_eval) / t
        ):
            current = candidato
            current_eval = candidato_eval

            # Atualiza a melhor solução global apenas quando há melhoria.
            if candidato_eval < best_eval:
                best = candidato.copy()
                best_eval = candidato_eval
                scores.append(best_eval)

    return best, best_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.2. SIMULATED ANNEALING — LIMITE DE AVALIAÇÕES
# ------------------------------------------------------------------------------
def simulated_annealing(objective, init_order, max_evals,
                        temp=1.0, seed=42):
    global avaliacoes

    rng = np.random.default_rng(seed)
    best = init_order.copy()
    best_eval = objective(best)
    current, current_eval = best.copy(), best_eval
    scores = [best_eval]
    t0 = time.perf_counter()
    step = 0

    # O contador inclui a avaliação inicial. O ciclo termina ao atingir
    # o orçamento máximo definido para a função objetivo.
    while avaliacoes < max_evals:
        t = temp / float(step + 1)
        candidato = current.copy()

        i, j = rng.choice(len(current), size=2, replace=False)
        candidato[i], candidato[j] = candidato[j], candidato[i]

        candidato_eval = objective(candidato)

        if candidato_eval < current_eval or rng.random() < math.exp(
            (current_eval - candidato_eval) / t
        ):
            current, current_eval = candidato, candidato_eval

            if candidato_eval < best_eval:
                best, best_eval = candidato.copy(), candidato_eval
                scores.append(best_eval)

        step += 1

    return best, best_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.3. TABU SEARCH — LIMITE DE ITERAÇÕES
# ------------------------------------------------------------------------------
def tabu_search_iterations(objective, init_order, max_iterations, tabu_tenure=7):
    current = init_order.copy()
    best = current.copy()

    current_eval = objective(current)
    best_eval = current_eval

    scores = [best_eval]
    tabu_list = []

    t0 = time.perf_counter()

    for _ in range(max_iterations):
        best_cand = None
        best_cand_eval = float("inf")
        best_move = None

        # Explora a vizinhança completa: todas as trocas possíveis (i, j),
        # com i < j. Para N=20 existem 190 movimentos.
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                move = (i, j)

                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                # A lista tabu guarda os pares de posições trocadas.
                # O critério de aspiração permite aceitar um movimento tabu
                # se este melhorar a melhor solução global encontrada.
                is_tabu = move in tabu_list
                aspiration = cand_eval < best_eval

                if (not is_tabu) or aspiration:
                    if cand_eval < best_cand_eval:
                        best_cand = cand
                        best_cand_eval = cand_eval
                        best_move = move

        if best_cand is None:
            break

        # Avança para o melhor vizinho admissível, mesmo que seja pior do que a solução corrente.
        current = best_cand
        current_eval = best_cand_eval

        # Atualização FIFO (First In First Out) da lista tabu, com dimensão máxima tabu_tenure.
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if current_eval < best_eval:
            best = current.copy()
            best_eval = current_eval
            scores.append(best_eval)

    return best, best_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.4. TABU SEARCH — LIMITE DE AVALIAÇÕES
# ------------------------------------------------------------------------------
def tabu_search(objective, init_order, max_evals, tabu_tenure=7):
    global avaliacoes

    current = init_order.copy()
    best = current.copy()

    current_eval = objective(current)
    best_eval = current_eval

    scores = [best_eval]
    tabu_list = []

    t0 = time.perf_counter()

    # Número de vizinhos da solução, correspondente a todas as trocas de duas posições distintas.
    num_vizinhos = len(current) * (len(current) - 1) // 2

    # Só inicia uma nova iteração se houver orçamento para avaliar a vizinhança completa.
    while avaliacoes + num_vizinhos <= max_evals:
        best_cand = None
        best_cand_eval = float("inf")
        best_move = None

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                move = (i, j)

                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                is_tabu = move in tabu_list
                aspiration = cand_eval < best_eval

                if (not is_tabu) or aspiration:
                    if cand_eval < best_cand_eval:
                        best_cand = cand
                        best_cand_eval = cand_eval
                        best_move = move

        if best_cand is None:
            break

        current = best_cand
        current_eval = best_cand_eval

        tabu_list.append(best_move)

        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if current_eval < best_eval:
            best = current.copy()
            best_eval = current_eval
            scores.append(best_eval)

    return best, best_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.5. FIRST IMPROVEMENT HILL CLIMBING — LIMITE DE ITERAÇÕES
# ------------------------------------------------------------------------------
def first_improvement_hill_climbing_iterations(objective, init_order,
                                               max_iterations):
    current = init_order.copy()
    current_eval = objective(current)
    scores = [current_eval]

    t0 = time.perf_counter()

    for _ in range(max_iterations):
        melhorou = False

        # Percorre a vizinhança e aceita imediatamente a primeira melhoria.
        # Ao encontrar uma, interrompe a exploração e inicia nova iteração.
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                if cand_eval < current_eval:
                    current = cand
                    current_eval = cand_eval
                    scores.append(current_eval)
                    melhorou = True
                    break

            if melhorou:
                break

        # Se nenhum vizinho melhorar a solução, foi atingido um mínimo local
        # relativamente à vizinhança definida pela operação swap.
        if not melhorou:
            break

    return current, current_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.6. FIRST IMPROVEMENT HILL CLIMBING — LIMITE DE AVALIAÇÕES
# ------------------------------------------------------------------------------
def first_improvement_hill_climbing_budget(objective, init_order, max_evals):
    global avaliacoes

    current = init_order.copy()
    current_eval = objective(current)
    scores = [current_eval]

    t0 = time.perf_counter()

    while avaliacoes < max_evals:
        melhorou = False
        passagem_completa = True

        for i in range(len(current)):
            for j in range(i + 1, len(current)):

                # Impede que o contador ultrapasse o orçamento definido.
                # Se o orçamento terminar a meio da vizinhança, a passagem
                # fica incompleta e não se pode garantir que foi encontrado
                # um mínimo local.
                if avaliacoes >= max_evals:
                    passagem_completa = False
                    break

                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                if cand_eval < current_eval:
                    current = cand
                    current_eval = cand_eval
                    scores.append(current_eval)
                    melhorou = True
                    break

            if melhorou or not passagem_completa:
                break

        if not melhorou:
            break

    return current, current_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.7. BEST IMPROVEMENT HILL CLIMBING — LIMITE DE ITERAÇÕES
# ------------------------------------------------------------------------------
def best_improvement_hill_climbing_iterations(objective, init_order, max_iterations):
    current = init_order.copy()
    current_eval = objective(current)
    scores = [current_eval]

    t0 = time.perf_counter()

    for _ in range(max_iterations):
        best_cand = None
        best_cand_eval = current_eval

        # Avalia toda a vizinhança e guarda o vizinho com menor custo.
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                if cand_eval < best_cand_eval:
                    best_cand = cand
                    best_cand_eval = cand_eval

        # Se nenhum vizinho melhorar a solução corrente, termina num mínimo local.
        if best_cand is None:
            break

        current = best_cand
        current_eval = best_cand_eval
        scores.append(current_eval)

    return current, current_eval, scores, time.perf_counter() - t0


# ------------------------------------------------------------------------------
# C.8. BEST IMPROVEMENT HILL CLIMBING — LIMITE DE AVALIAÇÕES
# ------------------------------------------------------------------------------
def best_improvement_hill_climbing_budget(objective, init_order, max_evals):
    global avaliacoes

    current = init_order.copy()
    current_eval = objective(current)
    scores = [current_eval]

    t0 = time.perf_counter()

    num_vizinhos = len(current) * (len(current) - 1) // 2

    # Tal como no Tabu Search, só começa uma iteração quando existe
    # orçamento suficiente para avaliar toda a vizinhança.
    while avaliacoes + num_vizinhos <= max_evals:
        best_cand = None
        best_cand_eval = current_eval

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]

                cand_eval = objective(cand)

                if cand_eval < best_cand_eval:
                    best_cand = cand
                    best_cand_eval = cand_eval

        if best_cand is None:
            break

        current = best_cand
        current_eval = best_cand_eval
        scores.append(current_eval)

    return current, current_eval, scores, time.perf_counter() - t0


# ==============================================================================
# D. EXPERIÊNCIA 1 — LIMITE MÁXIMO DE 500 ITERAÇÕES
# ==============================================================================
# Em cada seed, os quatro algoritmos recebem exatamente a mesma permutação
# inicial das 20 músicas. O contador de avaliações é reiniciado em cada execução.
n_iter = 500
seeds = range(10)

resultados_exp1 = []

for seed in seeds:
    ordem_inicial = np.random.default_rng(seed).permutation(N)

    algoritmos = {
        "Simulated Annealing": lambda: simulated_annealing_iterations(
            custo_ordem, ordem_inicial, n_iter, temp=1.0, seed=seed
        ),
        "Tabu Search": lambda: tabu_search_iterations(
            custo_ordem, ordem_inicial, n_iter, tabu_tenure=7
        ),
        "First Improvement HC": lambda: first_improvement_hill_climbing_iterations(
            custo_ordem, ordem_inicial, n_iter
        ),
        "Best Improvement HC": lambda: best_improvement_hill_climbing_iterations(
            custo_ordem, ordem_inicial, n_iter
        )
    }

    for nome, executar in algoritmos.items():
        avaliacoes = 0

        ordem_final, custo_final, historico, tempo = executar()

        resultados_exp1.append({
          "Seed": seed,
          "Algoritmo": nome,
          # O cálculo direto do custo inicial não incrementa o contador;
          # a avaliação inicial realizada dentro do algoritmo é contabilizada.
          "Custo inicial": C[ordem_inicial[:-1], ordem_inicial[1:]].sum(),
          "Custo final": custo_final,
          "Avaliações": avaliacoes,
          "Tempo (s)": tempo
        })

# Conversão dos resultados individuais para DataFrame.
df_exp1 = pd.DataFrame(resultados_exp1)

# Estatísticas agregadas das dez execuções de cada algoritmo.
resumo_exp1 = df_exp1.groupby("Algoritmo").agg(
    Custo_medio=("Custo final", "mean"),
    Custo_minimo=("Custo final", "min"),
    Custo_maximo=("Custo final", "max"),
    Desvio_padrao=("Custo final", "std"),
    Avaliacoes_medias=("Avaliações", "mean"),
    Tempo_medio=("Tempo (s)", "mean")
).round(4)

print("RESULTADOS POR EXECUÇÃO")
print(df_exp1.round(4).to_string(index=False))

print("\nRESUMO DA EXPERIÊNCIA 1")
print(resumo_exp1.to_string(index=False))


# ==============================================================================
# E. EXPERIÊNCIA 2 — ORÇAMENTO MÁXIMO DE 20 000 AVALIAÇÕES
# ==============================================================================
# Esta experiência limita o número de chamadas à função objetivo, permitindo
# comparar os algoritmos segundo um orçamento máximo de avaliações.
n_experiencias = 10
orcamento = 20000

resultados_experiencias_2 = []

for seed in range(n_experiencias):

    # A mesma seed produz a mesma solução inicial para os quatro algoritmos.
    ordem_inicial_teste = np.random.default_rng(seed).permutation(N)

    custo_inicial = C[
        ordem_inicial_teste[:-1],
        ordem_inicial_teste[1:]
    ].sum()

    algoritmos = {
        "Simulated Annealing": lambda: simulated_annealing(
            custo_ordem, ordem_inicial_teste, orcamento,
            temp=1.0, seed=seed
        ),

        "Tabu Search": lambda: tabu_search(
            custo_ordem, ordem_inicial_teste, orcamento,
            tabu_tenure=7
        ),

        "First Improvement HC": lambda: first_improvement_hill_climbing_budget(
            custo_ordem, ordem_inicial_teste, orcamento
        ),

        "Best Improvement HC": lambda: best_improvement_hill_climbing_budget(
            custo_ordem, ordem_inicial_teste, orcamento
        )
    }

    for nome, executar in algoritmos.items():

        # Reinicia o contador para que cada execução tenha o seu próprio orçamento.
        avaliacoes = 0

        _, custo_final, _, tempo = executar()

        resultados_experiencias_2.append({
            "Seed": seed,
            "Metodo": nome,
            "Custo_Inicial": custo_inicial,
            "Custo_Final": custo_final,
            "Avaliacoes": avaliacoes,
            "Tempo_s": tempo
        })


# Tabela com os resultados individuais das dez execuções.
df_experiencias_2 = pd.DataFrame(resultados_experiencias_2)

print("\nEXPERIÊNCIA 2: RESULTADOS DAS 10 EXECUÇÕES")
print(df_experiencias_2.round(4).to_string(index=False))


# Resumo estatístico: média, mínimo, máximo, desvio-padrão, avaliações
# médias e tempo médio para cada algoritmo.
resumo_2 = df_experiencias_2.groupby("Metodo").agg(
    Custo_Medio=("Custo_Final", "mean"),
    Custo_Minimo=("Custo_Final", "min"),
    Custo_Maximo=("Custo_Final", "max"),
    Desvio_Padrao=("Custo_Final", "std"),
    Avaliacoes_Medias=("Avaliacoes", "mean"),
    Tempo_Medio=("Tempo_s", "mean")
).reset_index()

print("\nRESUMO ESTATÍSTICO DA EXPERIÊNCIA 2")
print(resumo_2.round(4).to_string(index=False))
