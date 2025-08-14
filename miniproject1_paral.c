// gcc -O2 -fopenmp cross_sim.c -o cross_sim
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef enum { RED=0, YELLOW=1, GREEN=2 } LightState;
typedef enum { PH_NS_GREEN=0, PH_EW_GREEN=1 } Phase;   // NS = vertical, EW = horizontal

typedef struct {
    int pos;    // 0..N-1
    int lane;   // 0..L-1
    int alive;  // 1 activo
} Vehicle;

typedef struct {
    // Controlador de fases de la intersección (un solo nodo semafórico)
    Phase phase;
    int t_ns_green, t_ns_yellow;
    int t_ew_green, t_ew_yellow;
    int timer;
    LightState state_ns;  // luz que ve el flujo vertical
    LightState state_ew;  // luz que ve el flujo horizontal
} IntersectionCTL;

static inline int IDX(int lane, int pos, int N){ return lane*N + pos; }

static void clear_grid(int *occ, int size){
    #pragma omp parallel for schedule(static)
    for (int i=0;i<size;i++) occ[i] = -1;
}

static void ctl_step(IntersectionCTL *ctl){
    ctl->timer--;
    if (ctl->timer > 0) return;

    // Cambiar de estado completo (incluye amarillos)
    if (ctl->phase == PH_NS_GREEN){
        if (ctl->state_ns == GREEN){
            ctl->state_ns = YELLOW; ctl->state_ew = RED;
            ctl->timer = ctl->t_ns_yellow;
        } else {
            // pasó amarillo NS -> cambia a EW verde
            ctl->phase    = PH_EW_GREEN;
            ctl->state_ns = RED;
            ctl->state_ew = GREEN;
            ctl->timer    = ctl->t_ew_green;
        }
    } else { // PH_EW_GREEN
        if (ctl->state_ew == GREEN){
            ctl->state_ew = YELLOW; ctl->state_ns = RED;
            ctl->timer = ctl->t_ew_yellow;
        } else {
            // pasó amarillo EW -> cambia a NS verde
            ctl->phase    = PH_NS_GREEN;
            ctl->state_ns = GREEN;
            ctl->state_ew = RED;
            ctl->timer    = ctl->t_ns_green;
        }
    }
}

// Bloqueos por semáforo en la celda de intersección (parar EN la celda previa):
// - Horizontal: si quiere entrar a col_h y EW no está en VERDE -> bloquea
// - Vertical:   si quiere entrar a row_v y NS no está en VERDE -> bloquea
static inline int blocks_horizontal(const IntersectionCTL *ctl){
    return !(ctl->state_ew == GREEN); // amarillo cuenta como rojo
}
static inline int blocks_vertical(const IntersectionCTL *ctl){
    return !(ctl->state_ns == GREEN); // amarillo cuenta como rojo
}

// ------------------------ Movimiento general (paralelo) ------------------------
static int move_all_parallel_1D(Vehicle *veh, int numV,
                                int *occ_cur, int *occ_next,
                                int L, int N,
                                int inter_pos,                // posición de intersección en este eje
                                int (*should_block)(const IntersectionCTL*),
                                const IntersectionCTL *ctl,
                                omp_lock_t *locks)
{
    clear_grid(occ_next, L*N);
    int moved_count = 0;

    #pragma omp parallel for schedule(dynamic,64) reduction(+:moved_count)
    for (int i=0;i<numV;i++){
        if (!veh[i].alive) continue;

        int lane = veh[i].lane;
        int pos  = veh[i].pos;

        // salida del tramo
        if (pos == N-1){
            veh[i].alive = 0;
            continue;
        }

        int next_pos = pos + 1;

        // si el destino está ocupado en el estado actual, no avanza
        if (occ_cur[IDX(lane,next_pos,N)] != -1){
            // se queda
            omp_set_lock(&locks[IDX(lane,pos,N)]);
            occ_next[IDX(lane,pos,N)] = i;
            omp_unset_lock(&locks[IDX(lane,pos,N)]);
            continue;
        }

        // si el destino es la celda de intersección, aplica semáforo
        if (next_pos == inter_pos && should_block(ctl)){
            // está en rojo/amarillo para este flujo
            omp_set_lock(&locks[IDX(lane,pos,N)]);
            occ_next[IDX(lane,pos,N)] = i;
            omp_unset_lock(&locks[IDX(lane,pos,N)]);
            continue;
        }

        // competir por el destino en occ_next
        int dest = IDX(lane,next_pos,N);
        omp_set_lock(&locks[dest]);
        if (occ_next[dest] == -1){
            occ_next[dest] = i;
            veh[i].pos = next_pos;
            moved_count++;
            omp_unset_lock(&locks[dest]);
        } else {
            // alguien ya tomó la celda en next -> quedarse
            omp_unset_lock(&locks[dest]);
            int src = IDX(lane,pos,N);
            omp_set_lock(&locks[src]);
            occ_next[src] = i;
            omp_unset_lock(&locks[src]);
        }
    }
    return moved_count;
}

// ------------------------ Inicializaciones ------------------------
static void init_random_line(Vehicle *veh, int numV, int *occ, int L, int N, unsigned seed){
    srand(seed);
    clear_grid(occ, L*N);
    for (int i=0;i<numV;i++){
        int tries=0;
        while (1){
            int lane = rand()%L;
            int pos  = rand()%N;
            if (occ[IDX(lane,pos,N)] == -1){
                occ[IDX(lane,pos,N)] = i;
                veh[i].lane = lane;
                veh[i].pos = pos;
                veh[i].alive = 1;
                break;
            }
            if (++tries > 10*N){ veh[i].alive = 0; break; }
        }
    }
}

static void init_intersection(IntersectionCTL *ctl,
                              int t_ns_green, int t_ns_yel,
                              int t_ew_green, int t_ew_yel,
                              Phase start_phase)
{
    ctl->t_ns_green  = t_ns_green;
    ctl->t_ns_yellow = t_ns_yel;
    ctl->t_ew_green  = t_ew_green;
    ctl->t_ew_yellow = t_ew_yel;
    ctl->phase = start_phase;
    if (start_phase == PH_NS_GREEN){
        ctl->state_ns = GREEN; ctl->state_ew = RED;
        ctl->timer = t_ns_green;
    } else {
        ctl->state_ns = RED;   ctl->state_ew = GREEN;
        ctl->timer = t_ew_green;
    }
}

// ------------------------ MAIN: cruce en cruz ------------------------
int main(int argc, char **argv){
    // Defaults
    int Lh=2, Nh=100;          // horizontal: 2 carriles, 100 celdas
    int Lv=2, Nv=100;          // vertical:   2 carriles, 100 celdas
    int STEPS=200;
    int numVH=-1, numVV=-1;    // vehículos en H y V
    int col_h = 50;            // columna de intersección en H
    int row_v = 50;            // fila de intersección en V

    // CLI (opcional): Lh Nh Lv Nv STEPS numVH numVV col_h row_v
    if (argc>=2)  Lh     = atoi(argv[1]);
    if (argc>=3)  Nh     = atoi(argv[2]);
    if (argc>=4)  Lv     = atoi(argv[3]);
    if (argc>=5)  Nv     = atoi(argv[4]);
    if (argc>=6)  STEPS  = atoi(argv[5]);
    if (argc>=7)  numVH  = atoi(argv[6]);
    if (argc>=8)  numVV  = atoi(argv[7]);
    if (argc>=9)  col_h  = atoi(argv[8]);
    if (argc>=10) row_v  = atoi(argv[9]);

    if (numVH<0) numVH = (Lh*Nh)/6;   // densidad ≈ 1/6
    if (numVV<0) numVV = (Lv*Nv)/6;

    // Bounds sanos para el cruce
    if (col_h < 1) col_h = 1;
    if (col_h > Nh-2) col_h = Nh-2;
    if (row_v < 1) row_v = 1;
    if (row_v > Nv-2) row_v = Nv-2;

    // Memoria (H y V independientes)
    int *occH_cur  = (int*)malloc(sizeof(int)*Lh*Nh);
    int *occH_next = (int*)malloc(sizeof(int)*Lh*Nh);
    int *occV_cur  = (int*)malloc(sizeof(int)*Lv*Nv);
    int *occV_next = (int*)malloc(sizeof(int)*Lv*Nv);

    Vehicle *vehH = (Vehicle*)malloc(sizeof(Vehicle)*numVH);
    Vehicle *vehV = (Vehicle*)malloc(sizeof(Vehicle)*numVV);

    // Locks (uno por celda)
    omp_lock_t *locksH = (omp_lock_t*)malloc(sizeof(omp_lock_t)*Lh*Nh);
    omp_lock_t *locksV = (omp_lock_t*)malloc(sizeof(omp_lock_t)*Lv*Nv);
    for (int i=0;i<Lh*Nh;i++) omp_init_lock(&locksH[i]);
    for (int i=0;i<Lv*Nv;i++) omp_init_lock(&locksV[i]);

    // Init
    init_random_line(vehH, numVH, occH_cur, Lh, Nh, (unsigned)time(NULL));
    init_random_line(vehV, numVV, occV_cur, Lv, Nv, (unsigned)time(NULL)+1337);

    IntersectionCTL ctl;
    init_intersection(&ctl,
                      5, 2,   // NS: green, yellow
                      5, 2,   // EW: green, yellow
                      PH_NS_GREEN);

    // OpenMP
    omp_set_dynamic(1);
    omp_set_nested(1);

    for (int t=1; t<=STEPS; ++t){
        // Heurística de hilos: proporcional a carga total
        int totalV = numVH + numVV;
        int threads = totalV/32 + 1; if (threads<1) threads=1;
        omp_set_num_threads(threads);

        int movedH=0, movedV=0;

        // Avance en "secciones": controlador y dos flujos
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                ctl_step(&ctl);
            }
            #pragma omp section
            {
                movedH = move_all_parallel_1D(
                    vehH, numVH, occH_cur, occH_next, Lh, Nh,
                    col_h, blocks_horizontal, &ctl, locksH
                );
            }
            #pragma omp section
            {
                movedV = move_all_parallel_1D(
                    vehV, numVV, occV_cur, occV_next, Lv, Nv,
                    row_v, blocks_vertical, &ctl, locksV
                );
            }
        }

        // Swaps
        { int *tmp = occH_cur; occH_cur = occH_next; occH_next = tmp; }
        { int *tmp = occV_cur; occV_cur = occV_next; occV_next = tmp; }

        // Métricas: ocupación (vehículos activos en parrilla)
        long occH=0, occV=0;
        #pragma omp parallel for reduction(+:occH) schedule(static)
        for (int i=0;i<Lh*Nh;i++) occH += (occH_cur[i]!=-1);
        #pragma omp parallel for reduction(+:occV) schedule(static)
        for (int i=0;i<Lv*Nv;i++) occV += (occV_cur[i]!=-1);

        if (t%20==0){
            printf("Iter %3d | movedH=%d movedV=%d | occH=%ld occV=%ld | phase=%s (NS=%d, EW=%d) | threads=%d\n",
                   t, movedH, movedV, occH, occV,
                   (ctl.phase==PH_NS_GREEN?"NS_GREEN":"EW_GREEN"),
                   ctl.state_ns, ctl.state_ew,
                   omp_get_max_threads());
        }
    }

    // Cleanup
    for (int i=0;i<Lh*Nh;i++) omp_destroy_lock(&locksH[i]);
    for (int i=0;i<Lv*Nv;i++) omp_destroy_lock(&locksV[i]);
    free(locksH); free(locksV);
    free(occH_cur); free(occH_next); free(occV_cur); free(occV_next);
    free(vehH); free(vehV);
    return 0;
}
