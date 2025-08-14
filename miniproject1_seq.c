#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum { RED=0, YELLOW=1, GREEN=2 } LightState;

typedef struct {
    int pos;      // posición 0..N-1
    int lane;     // carril 0..L-1
    int alive;    // 1 activo
} Vehicle;

typedef struct {
    int position;     // celda que regula
    int t_red, t_yellow, t_green; // duración de cada estado
    int timer;        // contador restante del estado actual
    LightState state; // estado actual
} TrafficLight;

typedef struct {
    int L, N;
    int *occ; // matriz LxN linealizada: -1 libre, >=0 id de vehículo
} Road;

// ---- util
static inline int IDX(int lane, int pos, int N){ return lane*N + pos; }

static void road_clear(int *occ, int L, int N){
    for (int i=0;i<L*N;i++) occ[i] = -1;
}

// ---- luces
static void light_step(TrafficLight *tl){
    tl->timer--;
    if (tl->timer > 0) return;
    // rotar de estado
    if (tl->state == GREEN) { tl->state = YELLOW; tl->timer = tl->t_yellow; }
    else if (tl->state == YELLOW){ tl->state = RED; tl->timer = tl->t_red; }
    else { tl->state = GREEN; tl->timer = tl->t_green; }
}

static int light_blocks(const TrafficLight *tl, int cell_pos){
    // Bloquea si el vehículo intenta entrar EXACTAMENTE a la celda regulada y la luz está en rojo
    return (cell_pos == tl->position && tl->state == RED);
}

// ---- movimiento (1 celda máx por iteración)
static void move_all(Vehicle *veh, int numV, int *occ_cur, int *occ_next, int L, int N,
                     const TrafficLight *tls, int numTL){
    // limpiar next
    road_clear(occ_next, L, N);

    for (int i=0;i<numV;i++){
        if (!veh[i].alive) continue;
        int lane = veh[i].lane, pos = veh[i].pos;
        int next_pos = (pos+1 < N) ? pos+1 : pos; // tope al final
        // si ya está al final, se queda
        if (next_pos == pos){
            occ_next[IDX(lane,pos,N)] = i;
            continue;
        }
        // si la celda destino está ocupada en occ_cur, no avanza
        if (occ_cur[IDX(lane,next_pos,N)] != -1){
            occ_next[IDX(lane,pos,N)] = i;
            continue;
        }
        // chequear semáforos
        int blocked = 0;
        for (int k=0;k<numTL;k++){
            if (light_blocks(&tls[k], next_pos)){ blocked = 1; break; }
        }
        if (blocked){
            occ_next[IDX(lane,pos,N)] = i;
        } else {
            veh[i].pos = next_pos;
            occ_next[IDX(lane,next_pos,N)] = i;
        }
    }
}

// ---- init
static void init_random(Vehicle *veh, int numV, int *occ, int L, int N, unsigned seed){
    srand(seed);
    road_clear(occ, L, N);
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
            if (++tries > 10*N) { // fallback por seguridad
                veh[i].alive = 0; break;
            }
        }
    }
}

static void init_lights(TrafficLight *tl, int numTL, int N){
    // ejemplo: dos semáforos repartidos
    for (int i=0;i<numTL;i++){
        tl[i].position = (i+1)* (N/(numTL+1)); // distribución uniforme
        tl[i].t_red=3; tl[i].t_yellow=1; tl[i].t_green=3;
        tl[i].state = (i%2==0)?GREEN:RED;
        tl[i].timer = (tl[i].state==GREEN)?tl[i].t_green:tl[i].t_red;
    }
}

int main(int argc, char **argv){
    int L=3, N=120, numTL=2, STEPS=200, numV= (N*L)/6; // densidad inicial
    if (argc>=2) L = atoi(argv[1]);
    if (argc>=3) N = atoi(argv[2]);
    if (argc>=4) numTL = atoi(argv[3]);
    if (argc>=5) STEPS = atoi(argv[4]);
    if (argc>=6) numV = atoi(argv[5]);

    Road road = {L,N, NULL};
    road.occ = (int*)malloc(sizeof(int)*L*N);
    int *occ_next = (int*)malloc(sizeof(int)*L*N);

    Vehicle *veh = (Vehicle*)malloc(sizeof(Vehicle)*numV);
    TrafficLight *tls = (TrafficLight*)malloc(sizeof(TrafficLight)*numTL);

    init_lights(tls, numTL, N);
    init_random(veh, numV, road.occ, L, N, (unsigned)time(NULL));

    for (int t=1; t<=STEPS; ++t){
        // 1) actualizar semáforos
        for (int k=0;k<numTL;k++) light_step(&tls[k]);

        // 2) mover vehículos
        move_all(veh, numV, road.occ, occ_next, L, N, tls, numTL);

        // swap
        int *tmp = road.occ; road.occ = occ_next; occ_next = tmp;

        if (t%20==0){
            int moved=0;
            for (int i=0;i<numV;i++) if (veh[i].alive) moved += 1; // dummy métrica
            printf("Iter %3d | veh=%d, lights[0]=%d\n", t, numV, tls[0].state);
        }
    }

    free(road.occ); free(occ_next); free(veh); free(tls);
    return 0;
}
