function run() {
echo "$1 \n"

time ./CPP_BIO --data=$1 --data1=$2 --classification=true

echo "$1"
mv plot.ps $1.ps 
#read k
}

function launch() {
run $1 $2
}


launch data/EnoseAllSamples.csv data/SensoryAllSamples.csv 0
launch data/beef_fillets_enose.csv data/beef_fillets_enose_sensory.csv 0
launch data/beef_fillets_ftir.csv data/beef_fillets_ftir_sensory.csv 0
#launch data/beef_fillets_raman.csv.transposed.csv data/beef_fillets_raman_sensory.csv 0
launch data/minced_beef_ftir.csv.transposed.csv data/minced_beef_ftir_sensory.csv 0
launch data/minced_beef_hplc.csv data/minced_beef_hplc_sensory.csv 0
#pm-suspend
