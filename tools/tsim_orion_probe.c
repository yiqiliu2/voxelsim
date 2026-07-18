#include <stdio.h>
#include <stdlib.h>

#include "SIM_link.h"
#include "SIM_port.h"

int main(int argc, char **argv)
{
    double length_m = 1.0e-3;
    unsigned flit_bits = PARM_flit_width;

    if (argc > 1) {
        length_m = atof(argv[1]);
    }
    if (argc > 2) {
        flit_bits = (unsigned) atoi(argv[2]);
    }
    if (length_m <= 0.0 || flit_bits == 0) {
        fprintf(stderr, "usage: tsim_orion_probe [link_length_m] [flit_bits]\n");
        return 2;
    }

    double dyn_j_per_bit_per_m = LinkDynamicEnergyPerBitPerMeter(length_m, PARM_Vdd);
    double leakage_w_per_m = LinkLeakagePowerPerMeter(length_m, PARM_Vdd);
    double area_um2 = LinkArea(length_m, flit_bits);
    double dyn_j_per_flit = dyn_j_per_bit_per_m * length_m * (double) flit_bits;
    double leakage_w = leakage_w_per_m * length_m;

    printf("LinkDynamicEnergyPerFlitJ=%g\n", dyn_j_per_flit);
    printf("LinkLeakageW=%g\n", leakage_w);
    printf("LinkAreaUm2=%g\n", area_um2);
    return 0;
}
