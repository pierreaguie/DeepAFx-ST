
ssh pierre.aguie@requin.polytechnique.fr 
   export MASTER_ADDR=129.104.254.77
   export MASTER_PORT=12345
   export WORLD_SIZE=3
   export NODE_RANK=1
   export LOCAL_RANK=0

ssh pierre.aguie@raie.polytechnique.fr
   export MASTER_ADDR=129.104.254.77
   export MASTER_PORT=12345
   export WORLD_SIZE=3
   export NODE_RANK=2
   export LOCAL_RANK=0

ssh pierre.aguie@mulet.polytechnique.fr
   export MASTER_ADDR=129.104.254.77
   export MASTER_PORT=12345
   export WORLD_SIZE=3
   export NODE_RANK=0
   export LOCAL_RANK=0