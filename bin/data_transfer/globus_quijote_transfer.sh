#!/bin/bash
# bash script for running globus CLI to transfer Quijote Latin Hypercube HR ICs 


QUIJOTE_ENDPOINT="e0eae0aa-5bca-11ea-9683-0e56c063f437"
LONESTAR_ENDPOINT="fc2bdfc1-168f-4a55-8264-084b8c1b646c"

#globus login

globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/fc2bdfc1-168f-4a55-8264-084b8c1b646c/data_access]'

globus transfer $QUIJOTE_ENDPOINT:/ \
                $LONESTAR_ENDPOINT:/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr \
                --batch transfer_manifest.txt \
                --label "quijote initial conditions + rockstar halos" \
                --sync-level mtime
