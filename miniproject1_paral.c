#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

typedef enum { RED=0, YELLOW=1, GREEN=2 } LightState;

typedef struct {
    int pos;
    int lane;
    int alive;
} Vehicle;

typedef struct {
    int position;
    int t_red, t_yellow, t_green;
    int timer;
    LightState state;
} TrafficLight;

static inline int IDX(int lane, int pos, int N){ return lane*N + pos; }

static void road_clear(int *occ, int size){
    #pragma omp parallel for schedule(static)
    for (int i=0;i<size;i++) occ[i] = -1;
}

static void light_step(TrafficLight *tl){
    tl->timer--;
    if (tl->timer > 0) return;
    if (tl->state == GREEN) { tl->state = YELLOW; tl->timer = tl->t_yellow; }
    else if (tl->state == YELLOW){ tl->state = RED; tl->timer = tl->t_red; }
    else { tl->state = GREEN; tl->timer = tl->t_green; }
}

static int light_blocks(const TrafficLight *tl, int pos_actual, int pos_destino) {
    return (tl->state == RED &&
            pos_destino == tl->position &&
            pos_actual == tl->position - 1);
}

// Movimiento paralelo con locks por celda (en occ_next)
static int move_all_parallel(Vehicle *veh, int numV, int *occ_cur, int *occ_next,
                             int L, int N, const TrafficLight *tls, int numTL,
                             omp_lock_t *locks){
    // limpiar next
    road_clear(occ_next, L*N);

    int moved_count = 0;

    #pragma omp parallel for schedule(dynamic,64) reduction(+:moved_count)
    for (int i=0;i<numV;i++){
        if (!veh[i].alive) continue;
        int lane = veh[i].lane, pos = veh[i].pos;
        int next_pos = (pos+1 < N) ? pos+1 : pos;

        if (pos == N-1) {
            veh[i].alive = 0; // opcional: marcarlo como inactivo
            continue;         // no se coloca en occ_next, celda queda libre
        }
        if (next_pos == pos){
            // al final: sólo colocar
            omp_set_lock(&locks[IDX(lane,pos,N)]);
            occ_next[IDX(lane,pos,N)] = i;
            omp_unset_lock(&locks[IDX(lane,pos,N)]);
            continue;
        }
        // si está ocupada en occ_cur, no avanza
        if (occ_cur[IDX(lane,next_pos,N)] != -1){
            omp_set_lock(&locks[IDX(lane,pos,N)]);
            occ_next[IDX(lane,pos,N)] = i;
            omp_unset_lock(&locks[IDX(lane,pos,N)]);
            continue;
        }
        // chequear semáforos
        int blocked = 0;
        for (int k=0;k<numTL;k++){
            if (light_blocks(&tls[k], pos, next_pos)) { blocked = 1; break; }
        }
        
        if (blocked){
            omp_set_lock(&locks[IDX(lane,pos,N)]);
            occ_next[IDX(lane,pos,N)] = i;
            omp_unset_lock(&locks[IDX(lane,pos,N)]);
        } else {
            // competir por la celda destino en occ_next
            int dest = IDX(lane,next_pos,N);
            omp_set_lock(&locks[dest]);
            if (occ_next[dest] == -1){
                occ_next[dest] = i;
                veh[i].pos = next_pos;
                moved_count++;
                omp_unset_lock(&locks[dest]);
            } else {
                // ya tomado por otro: quedarse
                omp_unset_lock(&locks[dest]);
                int src = IDX(lane,pos,N);
                omp_set_lock(&locks[src]);
                occ_next[src] = i;
                omp_unset_lock(&locks[src]);
            }
        }
    }
    return moved_count;
}

static void init_random(Vehicle *veh, int numV, int *occ, int L, int N, unsigned seed){
    srand(seed);
    for (int i=0;i<L*N;i++) occ[i] = -1;
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

static void init_lights(TrafficLight *tl, int numTL, int N){
    for (int i=0;i<numTL;i++){
        tl[i].position = (i+1)* (N/(numTL+1));
        tl[i].t_red=3; tl[i].t_yellow=1; tl[i].t_green=3;
        tl[i].state = (i%2==0)?GREEN:RED;
        tl[i].timer = (tl[i].state==GREEN)?tl[i].t_green:tl[i].t_red;
    }
}

int main(int argc, char **argv){
    int L=3, N=120, numTL=2, STEPS=200, numV=-1;
    if (argc>=2) L = atoi(argv[1]);
    if (argc>=3) N = atoi(argv[2]);
    if (argc>=4) numTL = atoi(argv[3]);
    if (argc>=5) STEPS = atoi(argv[4]);
    if (argc>=6) numV = atoi(argv[5]);
    if (numV<0) numV = (N*L)/6;

    int *occ_cur  = (int*)malloc(sizeof(int)*L*N);
    int *occ_next = (int*)malloc(sizeof(int)*L*N);
    Vehicle *veh = (Vehicle*)malloc(sizeof(Vehicle)*numV);
    TrafficLight *tls = (TrafficLight*)malloc(sizeof(TrafficLight)*numTL);

    init_lights(tls, numTL, N);
    init_random(veh, numV, occ_cur, L, N, (unsigned)time(NULL));

    // ---- OpenMP setup (dinámico + anidado)
    omp_set_dynamic(1);     // permitir ajuste dinámico
    omp_set_nested(1);      // habilitar paralelismo anidado (opcional)

    // locks por celda para occ_next
    omp_lock_t *locks = (omp_lock_t*)malloc(sizeof(omp_lock_t)*L*N);
    for (int i=0;i<L*N;i++) omp_init_lock(&locks[i]);

    for (int t=1; t<=STEPS; ++t){
        // ajustar #hilos según carga (ejemplo simple)
        int threads = numV/32 + 1; if (threads<1) threads=1;
        omp_set_num_threads(threads);

        int moved=0;

        // ------- SECTIONS: luces y movimiento en paralelo -------
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // (Se podría paralelizar por semáforo si numTL es grande)
                for (int k=0;k<numTL;k++) light_step(&tls[k]);
            }
            #pragma omp section
            {
                moved = move_all_parallel(veh, numV, occ_cur, occ_next, L, N, tls, numTL, locks);
            }
        }

        // swap buffers
        int *tmp = occ_cur; occ_cur = occ_next; occ_next = tmp;

        // ---- sección de métricas (demostración de paralelismo anidado)
        long occ_count=0;
        #pragma omp parallel for reduction(+:occ_count) schedule(static)
        for (int i=0;i<L*N;i++) occ_count += (occ_cur[i] != -1);

        if (t%20==0){
            printf("Iter %3d | moved=%d | occ=%ld | light0=%d | threads=%d\n",
                   t, moved, occ_count, (numTL>0?tls[0].state:-1), omp_get_max_threads());
        }
    }

    for (int i=0;i<L*N;i++) omp_destroy_lock(&locks[i]);
    free(locks);
    free(occ_cur); free(occ_next); free(veh); free(tls);
    return 0;
}
