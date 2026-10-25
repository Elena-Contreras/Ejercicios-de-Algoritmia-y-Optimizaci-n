import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("=== EJERCICIO 1: Parallel Speedup (estimate_pi) ===")
    seeds = list(range(8))  

    t0 = time.perf_counter()
    seq_results = [estimate_pi(s) for s in seeds]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        par_results = pool.map(estimate_pi, seeds)
    t_par = time.perf_counter() - t0

    print(f"Sequential wall-clock: {t_seq:.3f} s")
    print(f"Parallel wall-clock:   {t_par:.3f} s (Speedup: {t_seq/t_par:.2f}x)")
    print(f"pi estimate (mean of {len(seeds)} runs): {sum(par_results) / len(par_results):.5f}")

    print()
    print(f"Nucleos del CPU: {mp.cpu_count()}  (8 tareas de 2,000,000 muestras cada una)")
    workers = [1, 2, 4, 8]
    t_proc, t_hilos = [], []
    tarea = partial(estimate_pi, samples=2_000_000)
    for p in workers:
        t0 = time.perf_counter()
        with mp.Pool(processes=p) as pool:               
            pool.map(tarea, seeds)
        t_proc.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as ex: 
            list(ex.map(tarea, seeds))
        t_hilos.append(time.perf_counter() - t0)
    sp_proc = [t_proc[0] / t for t in t_proc]
    sp_hilos = [t_hilos[0] / t for t in t_hilos]
    print(f"{'Workers':>7} | {'Procesos (s)':>12} {'Speedup':>8} {'Eficiencia':>11} | {'Hilos (s)':>10} {'Speedup':>8}")
    for i, p in enumerate(workers):
        print(f"{p:>7} | {t_proc[i]:>12.3f} {sp_proc[i]:>8.2f} {sp_proc[i] / p:>11.2f} | {t_hilos[i]:>10.3f} {sp_hilos[i]:>8.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(workers, t_proc, "o-", color="tab:green", label="multiprocessing")
    axs[0].plot(workers, t_hilos, "s-", color="tab:red", label="threading (GIL)")
    axs[0].set_xlabel("Numero de workers")
    axs[0].set_ylabel("Tiempo (s)")
    axs[0].set_title("Tiempo vs workers")
    axs[0].set_xticks(workers)
    axs[0].legend()
    axs[1].plot(workers, workers, "k--", label="Speedup ideal (lineal)")
    axs[1].plot(workers, sp_proc, "o-", color="tab:green", label="multiprocessing")
    axs[1].plot(workers, sp_hilos, "s-", color="tab:red", label="threading (GIL)")
    axs[1].set_xlabel("Numero de workers")
    axs[1].set_ylabel("Speedup = T(1) / T(p)")
    axs[1].set_title("Speedup vs workers")
    axs[1].set_xticks(workers)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    print()
    print("=== EJERCICIO 2: Analisis algoritmico (mochila voraz) ===")

    def knapsack(values: list[int], weights: list[int], W: int):
        n = len(values)
        K = [[0] * (W + 1) for _ in range(n + 1)]   
        for i in range(1, n + 1):
            v, w = values[i - 1], weights[i - 1]
            for c in range(W + 1):
                K[i][c] = K[i - 1][c]             
                if w <= c:
                    K[i][c] = max(K[i][c], v + K[i - 1][c - w])      
        chosen, c = [], W                  
        for i in range(n, 0, -1):
            if K[i][c] != K[i - 1][c]:
                chosen.append(i - 1)
                c -= weights[i - 1]
        return K, K[n][W], chosen[::-1]

    items, values, weights = ["A", "B", "C"], [60, 100, 120], [10, 20, 30]
    K, best, chosen = knapsack(values, weights, 50)
    print("capacity:", *[f"{c:4d}" for c in range(0, 51, 10)])
    for i, row in enumerate(K):
        print(f"{'-' if i == 0 else '+' + items[i - 1]:>8s}:", *[f"{row[c]:4d}" for c in range(0, 51, 10)])
    print(f"Best value = {best} with items {[items[i] for i in chosen]}  (greedy found 160)")

    def greedy_menor_peso(values, weights, W):
        orden = sorted(range(len(values)), key=lambda i: weights[i])
        elegidos, peso, valor = [], 0, 0
        for i in orden:
            if peso + weights[i] <= W:
                elegidos.append(i)
                peso += weights[i]
                valor += values[i]
        return elegidos, valor, peso

    def greedy_razon(values, weights, W):
        orden = sorted(range(len(values)), key=lambda i: values[i] / weights[i], reverse=True)
        elegidos, peso, valor = [], 0, 0
        for i in orden:
            if peso + weights[i] <= W:
                elegidos.append(i)
                peso += weights[i]
                valor += values[i]
        return elegidos, valor, peso

    def comparar(titulo, nombres, vals, pesos, W):
        print(titulo)
        print(f"Capacidad W = {W}")
        print(f"{'Item':>5} {'Valor v':>8} {'Peso w':>7} {'v/w':>7}")
        for nom, v, w in zip(nombres, vals, pesos):
            print(f"{nom:>5} {v:>8} {w:>7} {v / w:>7.2f}")
        c1, v1, w1 = greedy_menor_peso(vals, pesos, W)
        c2, v2, w2 = greedy_razon(vals, pesos, W)
        _, optimo, _ = knapsack(vals, pesos, W)   
        print(f"Menor peso primero : items {[nombres[i] for i in c1]}  valor = {v1}  peso usado = {w1}")
        print(f"Razon v/w          : items {[nombres[i] for i in c2]}  valor = {v2}  peso usado = {w2}")
        print(f"Optimo (prog. din.): valor = {optimo}")
        if v2 > v1:
            print(f"--> La razon v/w gana por {v2 - v1} unidades de valor.")
        elif v2 == v1:
            print("--> Empatan en esta instancia.")
        else:
            print("--> Aqui gana menor peso.")
        return [("Menor peso", [nombres[i] for i in c1], v1, w1),
                ("Razon v/w", [nombres[i] for i in c2], v2, w2)], optimo

    print()
    res1, opt1 = comparar("Instancia 1 (ejemplo base):", ["A", "B", "C"], [60, 100, 120], [10, 20, 30], 50)
    print()
    assert all(v / w > w for v, w in zip([3, 12, 100], [1, 3, 9]))  
    res2, opt2 = comparar("Instancia 2 (nueva, todos los items cumplen v/w > w):",
                          ["A", "B", "C"], [3, 12, 100], [1, 3, 9], 10)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for fila, (titulo, W, res, optimo) in enumerate([("Instancia 1", 50, res1, opt1),
                                                    ("Instancia 2", 10, res2, opt2)]):
        etiquetas = [r[0] for r in res]
        colores = ["tab:red", "tab:green"]
        barras = axs[fila][0].bar(etiquetas, [r[2] for r in res], color=colores)
        for b, r in zip(barras, res):
            axs[fila][0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                              f"valor {r[2]}\nitems: {', '.join(r[1])}", ha="center", va="bottom", fontsize=9)
        axs[fila][0].axhline(optimo, color="gray", linestyle="--", label=f"Optimo = {optimo}")
        axs[fila][0].set_ylim(0, optimo * 1.3)
        axs[fila][0].set_ylabel("Valor total")
        axs[fila][0].set_title(f"{titulo}: valor total obtenido")
        axs[fila][0].legend(loc="lower right")
        barras = axs[fila][1].bar(etiquetas, [r[3] for r in res], color=colores)
        for b, r in zip(barras, res):
            axs[fila][1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                              f"{r[3]} de {W}", ha="center", va="bottom", fontsize=9)
        axs[fila][1].axhline(W, color="k", linestyle="--", label=f"Capacidad W = {W}")
        axs[fila][1].set_ylim(0, W * 1.25)
        axs[fila][1].set_ylabel("Capacidad usada")
        axs[fila][1].set_title(f"{titulo}: capacidad usada")
        axs[fila][1].legend(loc="lower right")
    plt.show()

    print()

    print("=== EJERCICIO 3: Busqueda local (2-opt para el TSP) ===")
    import math

    cities = {"A": (0, 0), "B": (0, 3), "E": (2, 5), "D": (4, 3), "C": (4, 0)}

    def tour_length(route: list[str]) -> float:
        return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
                   for i in range(len(route)))

    def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
        """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
        return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

    route = ["A", "D", "E", "B", "C"]
    new_route = two_opt_move(route, 1, 3)
    print(f"Before: {route}  L = {tour_length(route):.2f}")
    print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")

    def subir_colinas_2opt(route):
        mejor = route[:]
        mejor_len = tour_length(mejor)
        mejoras = 0
        hay_mejora = True
        while hay_mejora:
            hay_mejora = False
            for i in range(1, len(mejor) - 1):
                for k in range(i + 1, len(mejor)):
                    candidata = two_opt_move(mejor, i, k)
                    L = tour_length(candidata)
                    if L < mejor_len - 1e-9:
                        mejor, mejor_len = candidata, L
                        mejoras += 1
                        hay_mejora = True
        return mejor, mejor_len, mejoras

    def vecino_mas_cercano(inicio):
        ruta = [inicio]
        faltan = set(cities) - {inicio}
        while faltan:
            sig = min(faltan, key=lambda c: math.dist(cities[ruta[-1]], cities[c]))
            ruta.append(sig)
            faltan.remove(sig)
        return ruta

    print()
    nombres = list("ABCDEFGHIJ")       
    rng_c = random.Random(3)
    cities = {c: (rng_c.uniform(0, 100), rng_c.uniform(0, 100)) for c in nombres}
    rng = random.Random(3)

    # Parte 1: ruta inicial al azar
    ruta_azar = nombres[:]
    rng.shuffle(ruta_azar)
    L0 = tour_length(ruta_azar)
    t0 = time.perf_counter()
    mejor1, L1, mej1 = subir_colinas_2opt(ruta_azar)
    t1 = time.perf_counter() - t0

    inicio = nombres[rng.randrange(10)]
    ruta_nn = vecino_mas_cercano(inicio)
    L0nn = tour_length(ruta_nn)
    t0 = time.perf_counter()
    mejor2, L2, mej2 = subir_colinas_2opt(ruta_nn)
    t2 = time.perf_counter() - t0

    print("Parte 1 - inicio al azar:")
    print(f"  Ruta inicial : {ruta_azar}  L = {L0:.2f}")
    print(f"  Mejor ruta   : {mejor1}  L = {L1:.2f}")
    print(f"  Mejoras (movimientos 2-opt aceptados) = {mej1}   tiempo = {t1 * 1000:.3f} ms")
    print()
    print(f"Parte 2 - inicio con vecino mas cercano (empezando en el nodo {inicio}):")
    print(f"  Ruta inicial : {ruta_nn}  L = {L0nn:.2f}")
    print(f"  Mejor ruta   : {mejor2}  L = {L2:.2f}")
    print(f"  Mejoras (movimientos 2-opt aceptados) = {mej2}   tiempo = {t2 * 1000:.3f} ms")
    print()
    print("Comparacion de esta corrida:")
    print(f"  Mejoras: azar = {mej1}  vs  vecino mas cercano = {mej2}")
    print(f"  Distancia final: azar = {L1:.2f}  vs  vecino mas cercano = {L2:.2f}")
    print(f"  Tiempo: azar = {t1 * 1000:.3f} ms  vs  vecino mas cercano = {t2 * 1000:.3f} ms")

    print()
    print("Repeticion con 200 instancias aleatorias de 10 nodos (para no depender de la suerte):")
    R = 200
    mej_a, mej_n, fin_a, fin_n, t_a, t_n = [], [], [], [], [], []
    for r in range(R):
        rc = random.Random(1000 + r)
        cities = {c: (rc.uniform(0, 100), rc.uniform(0, 100)) for c in nombres}
        rr = random.Random(r)
        a = nombres[:]
        rr.shuffle(a)
        b = vecino_mas_cercano(nombres[rr.randrange(10)])
        t0 = time.perf_counter()
        _, La, ma = subir_colinas_2opt(a)
        t_a.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _, Lb, mb = subir_colinas_2opt(b)
        t_n.append(time.perf_counter() - t0)
        mej_a.append(ma)
        mej_n.append(mb)
        fin_a.append(La)
        fin_n.append(Lb)
    prom = lambda x: sum(x) / len(x)
    menos = sum(mb < ma for ma, mb in zip(mej_a, mej_n))
    print(f"  Mejoras promedio : azar = {prom(mej_a):.2f}  vs  vecino mas cercano = {prom(mej_n):.2f}")
    print(f"  Distancia final promedio: azar = {prom(fin_a):.2f}  vs  vecino mas cercano = {prom(fin_n):.2f}")
    print(f"  Tiempo promedio  : azar = {prom(t_a) * 1000:.3f} ms  vs  vecino mas cercano = {prom(t_n) * 1000:.3f} ms")
    print(f"  En {menos} de {R} instancias el vecino mas cercano necesito menos mejoras que el azar.")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    etiq = ["Inicio aleatorio", "Inicio vecino\nmas cercano"]
    axs[0].boxplot([mej_a, mej_n], showmeans=True)
    axs[0].set_xticks([1, 2])
    axs[0].set_xticklabels(etiq)
    axs[0].set_ylabel("Numero de mejoras 2-opt")
    axs[0].set_title("Mejoras hasta el optimo local (200 instancias)")
    axs[1].boxplot([fin_a, fin_n], showmeans=True)
    axs[1].set_xticks([1, 2])
    axs[1].set_xticklabels(etiq)
    axs[1].set_ylabel("Distancia del optimo local alcanzado")
    axs[1].set_title("Calidad del optimo local (200 instancias)")
    plt.tight_layout()
    plt.show()

    print()

    print("=== EJERCICIO 4: Ajuste de parametros (Recocido Simulado) ===")
    import random

    def simulated_annealing(initial, schedule, max_iter, rng, t_min=1e-3):
        current = initial[:]
        cur_value = -tour_length(current)       
        best, best_len = current[:], -cur_value
        hist_obj, hist_T, malos = [], [], 0
        for t in range(1, max_iter + 1):
            T = schedule(t)
            if T < t_min:    
                break
            i, k = sorted(rng.sample(range(1, len(current)), 2))
            nxt = two_opt_move(current, i, k)   
            next_value = -tour_length(nxt)
            delta_E = next_value - cur_value
            if delta_E > 0:
                current, cur_value = nxt, next_value
            elif rng.random() < math.exp(delta_E / T):  
                current, cur_value = nxt, next_value
                malos += 1
            if -cur_value < best_len:
                best, best_len = current[:], -cur_value
            hist_obj.append(-cur_value)
            hist_T.append(T)
        return best, best_len, hist_obj, hist_T, malos

    N, T0 = 10_000, 100.0
    rng_c = random.Random(4)
    cities = {f"C{i}": (rng_c.uniform(0, 100), rng_c.uniform(0, 100)) for i in range(30)}  
    inicial = list(cities)
    random.Random(4).shuffle(inicial)               
    print(f"Distancia inicial (ruta al azar): {tour_length(inicial):.2f}")
    print(f"T0 = {T0}, maximo de iteraciones = {N}, se detiene si T < 0.001")

    esquemas = {
        "Exponencial rapido (T0*0.99^t)": lambda t: T0 * 0.99 ** t,
        "Lineal (T0*(1-t/N))": lambda t: T0 * (1 - t / N),
        "Exponencial lento (T0*0.999^t)": lambda t: T0 * 0.999 ** t,
    }
    colores = ["tab:blue", "tab:orange", "tab:green"]
    historiales = {}
    print(f"{'Esquema':<32} {'Mejor dist.':>11} {'Iteraciones':>12} {'Tiempo (s)':>11} {'Malos aceptados':>16}")
    for nombre, sch in esquemas.items():
        rng_sa = random.Random(42)
        t0 = time.perf_counter()
        mejor, mejor_len, h_obj, h_T, malos = simulated_annealing(inicial, sch, N, rng_sa)
        dt = time.perf_counter() - t0
        historiales[nombre] = (h_obj, h_T)
        print(f"{nombre:<32} {mejor_len:>11.2f} {len(h_T):>12} {dt:>11.3f} {malos:>16}")

    REP = 20
    resumen = {}
    for nombre, sch in esquemas.items():
        tiempos_r, calidades_r = [], []
        for semilla in range(REP):
            t0 = time.perf_counter()
            _, L_fin, _, _, _ = simulated_annealing(inicial, sch, N, random.Random(semilla))
            tiempos_r.append((time.perf_counter() - t0) * 1000)
            calidades_r.append(L_fin)
        resumen[nombre] = (tiempos_r, calidades_r)
    print()
    print(f"Promedio de {REP} corridas (semillas distintas) por esquema:")
    print(f"{'Esquema':<32} {'Dist. prom.':>11} {'Tiempo prom. (ms)':>18}")
    for nombre, (tr, cr) in resumen.items():
        print(f"{nombre:<32} {prom(cr):>11.2f} {prom(tr):>18.1f}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for (nombre, (h_obj, h_T)), col in zip(historiales.items(), colores):
        axs[0].plot(range(1, len(h_T) + 1), h_T, color=col, label=nombre)
        axs[1].plot(range(1, len(h_obj) + 1), h_obj, color=col, label=nombre, alpha=0.7)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-3, T0 * 2)
    axs[0].set_xlabel("Iteraciones")
    axs[0].set_ylabel("Temperatura (escala log)")
    axs[0].set_title("Esquemas de enfriamiento")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("Iteraciones")
    axs[1].set_ylabel("Costo (distancia del recorrido)")
    axs[1].set_title("Curvas de convergencia: costo vs iteraciones")
    axs[1].legend(fontsize=8)
    for (nombre, (tr, cr)), col in zip(resumen.items(), colores):
        axs[2].scatter(tr, cr, color=col, alpha=0.35, s=20)
        axs[2].scatter(prom(tr), prom(cr), color=col, edgecolor="k", s=130, marker="D", label=nombre.split(" (")[0])
    axs[2].set_xlabel("Tiempo de ejecucion (ms)")
    axs[2].set_ylabel("Distancia final (calidad)")
    axs[2].set_title(f"Tiempo vs calidad ({REP} corridas; rombo = promedio)")
    axs[2].legend(fontsize=8, loc="upper center")
    plt.tight_layout()
    plt.show()

    print()
    print("=== EJERCICIO 5: Razon de aproximacion (MST) ===")
    import itertools
    import math
    import random

    def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
        best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
        children = {v: [] for v in range(len(points))}
        while best:
            v = min(best, key=lambda u: best[u][0])
            children[best.pop(v)[1]].append(v)     
            for u in best:
                d = math.dist(points[v], points[u])
                if d < best[u][0]:
                    best[u] = (d, v)
        return children

    def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
        return [root] + [c for child in children[root] for c in preorder(children, child)]

    def tour_length(points, tour) -> float:
        return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

    random.seed(7)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(9)]

    approx = preorder(mst_prim(points))
    optimal = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
                  key=lambda t: tour_length(points, t))  

    print(f"MST approximation: {approx}  L = {tour_length(points, approx):.2f}")
    print(f"Optimal (brute):   {list(optimal)}  L = {tour_length(points, optimal):.2f}")
    print(f"Ratio = {tour_length(points, approx) / tour_length(points, optimal):.3f}  (guaranteed <= 2)")

    def peso_arbol(points, children):
        return sum(math.dist(points[padre], points[hijo]) for padre in children for hijo in children[padre])

    def optimo_bruto(points):
        opt = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
                  key=lambda t: tour_length(points, t))
        return tour_length(points, opt)

    def subir_colinas_2opt_pts(points, tour):
        mejor = list(tour)
        mejor_len = tour_length(points, mejor)
        mejoras = 0
        hay_mejora = True
        while hay_mejora:
            hay_mejora = False
            for i in range(1, len(mejor) - 1):
                for k in range(i + 1, len(mejor)):
                    candidata = two_opt_move(mejor, i, k)
                    L = tour_length(points, candidata)
                    if L < mejor_len - 1e-9:
                        mejor, mejor_len = candidata, L
                        mejoras += 1
                        hay_mejora = True
        return mejor, mejor_len, mejoras

    def vecino_pts(points, inicio):
        ruta = [inicio]
        faltan = set(range(len(points))) - {inicio}
        while faltan:
            sig = min(faltan, key=lambda c: math.dist(points[ruta[-1]], points[c]))
            ruta.append(sig)
            faltan.remove(sig)
        return ruta

    def recocido_pts(points, inicial, rng, T0=10.0, alpha=0.995, t_min=1e-3):
        actual = inicial[:]
        L_act = tour_length(points, actual)
        mejor, L_mejor = actual[:], L_act
        T = T0
        while T >= t_min:
            i, k = sorted(rng.sample(range(1, len(actual)), 2))
            cand = two_opt_move(actual, i, k)
            L_c = tour_length(points, cand)
            delta_E = L_act - L_c
            if delta_E > 0 or rng.random() < math.exp(delta_E / T):
                actual, L_act = cand, L_c
                if L_act < L_mejor:
                    mejor, L_mejor = actual[:], L_act
            T *= alpha
        return mejor, L_mejor

    print()
    print("100 instancias aleatorias de 9 ciudades (la fuerza bruta tarda un rato, paciencia...)")
    R = 100
    razon_mst, razon_mst2, razon_azar2, razon_nn2, razon_sa = [], [], [], [], []
    mej_mst, mej_azar, mej_nn = [], [], []
    cota_ok = opt_ok = 0
    for r in range(R):
        rc = random.Random(500 + r)
        pts = [(rc.uniform(0, 10), rc.uniform(0, 10)) for _ in range(9)]
        arbol = mst_prim(pts)
        w_mst = peso_arbol(pts, arbol)    
        ruta = preorder(arbol)
        L_apr = tour_length(pts, ruta)
        L_opt = optimo_bruto(pts)
        cota = 2 * w_mst       
        cota_ok += L_apr <= cota + 1e-9
        opt_ok += L_opt >= w_mst - 1e-9

        rr = random.Random(r)
        lista = list(range(9))  
        rr.shuffle(lista)
        inicio_nn = rr.randrange(9)

        _, L_m2, mej_m = subir_colinas_2opt_pts(pts, ruta)          
        _, L_r2, mej_r = subir_colinas_2opt_pts(pts, lista)   
        _, L_n2, mej_n2 = subir_colinas_2opt_pts(pts, vecino_pts(pts, inicio_nn))   
        _, L_sa = recocido_pts(pts, lista, random.Random(10_000 + r))   

        razon_mst.append(L_apr / L_opt)
        razon_mst2.append(L_m2 / L_opt)
        razon_azar2.append(L_r2 / L_opt)
        razon_nn2.append(L_n2 / L_opt)
        razon_sa.append(L_sa / L_opt)
        mej_mst.append(mej_m)
        mej_azar.append(mej_r)
        mej_nn.append(mej_n2)

    print(f"Se cumple  L(aproximacion) <= 2 x MST  en {cota_ok} de {R} instancias.")
    print(f"Se cumple  MST <= optimo               en {opt_ok} de {R} instancias.")
    print()
    print("Razon  C_obtenido / C_optimo:")
    print(f"{'Metodo':<34} {'Promedio':>9} {'Maximo':>8} {'Minimo':>8} {'Mejoras prom.':>14}")
    filas = [("MST puro (preorden)", razon_mst, None),
             ("MST + 2-opt", razon_mst2, mej_mst),
             ("Busqueda local (inicio azar)", razon_azar2, mej_azar),
             ("Busqueda local (inicio vec. cercano)", razon_nn2, mej_nn),
             ("Recocido simulado", razon_sa, None)]
    for nombre, datos, mejs in filas:
        extra = f"{prom(mejs):>14.2f}" if mejs is not None else f"{'-':>14}"
        print(f"{nombre:<34} {prom(datos):>9.3f} {max(datos):>8.3f} {min(datos):>8.3f} {extra}")
    print()
    print(f"La razon maxima del MST fue {max(razon_mst):.3f}; la garantia teorica es 2.")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(razon_mst, bins=[1 + 0.05 * i for i in range(23)], color="tab:blue", alpha=0.75, label="MST puro")
    axs[0].hist(razon_mst2, bins=[1 + 0.05 * i for i in range(23)], color="tab:green", alpha=0.6, label="MST + 2-opt")
    axs[0].axvline(2, color="red", linestyle="--", label="Limite teorico = 2")
    axs[0].set_xlim(0.98, 2.1)
    axs[0].set_xlabel("Razon de aproximacion  C_obtenido / C_optimo")
    axs[0].set_ylabel("Numero de instancias")
    axs[0].set_title("Histograma de la razon de aproximacion")
    axs[0].legend(fontsize=8)
    x = range(1, R + 1)
    axs[1].scatter(x, razon_mst, s=14, color="tab:blue", label="MST puro")
    axs[1].scatter(x, razon_mst2, s=14, color="tab:green", marker="s", label="MST + 2-opt")
    axs[1].axhline(2, color="red", linestyle="--", label="Limite teorico = 2")
    axs[1].axhline(1, color="k", linestyle=":", label="Optimo = 1")
    axs[1].set_ylim(0.95, 2.1)
    axs[1].set_xlabel("Instancia")
    axs[1].set_ylabel("Razon  C_obtenido / C_optimo")
    axs[1].set_title("Dispersion de la razon en 100 instancias")
    axs[1].legend(fontsize=8)
    axs[2].boxplot([razon_mst, razon_mst2, razon_azar2, razon_nn2, razon_sa], showmeans=True)
    axs[2].set_xticks([1, 2, 3, 4, 5])
    axs[2].set_xticklabels(["MST\npuro", "MST +\n2-opt", "Busq. local\n(azar)", "Busq. local\n(vec. cercano)", "Recocido\nsimulado"], fontsize=8)
    axs[2].axhline(1, color="k", linestyle=":")
    axs[2].set_ylabel("Costo obtenido / costo optimo")
    axs[2].set_title("Comparacion de metodos (100 instancias)")
    plt.tight_layout()
    plt.show()