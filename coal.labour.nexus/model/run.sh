#/bin/bash
# File to launch several coal labour nexus with Scilab

if [ $HOSTNAME = "poseidon.centre-cired.fr" ] || [ $HOSTNAME = "belenus.centre-cired.fr" ]
then
    scilabExe='/home/bibas/bin/scilab-5.4.1/bin/scilab'
elif [ $HOSTNAME = "inari.centre-cired.fr" ]
then
    scilabExe='/data/software/scilab-5.4.1/bin/scilab'
else
    scilabExe='scilab'
fi


echo "nohup nice $scilabExe -nb -nwni -f V2_batch.labour.sce > /dev/null 2> run.batch.err < /dev/null"            > run.batchCmd
sh run.batchCmd &