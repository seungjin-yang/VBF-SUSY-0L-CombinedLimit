function run-combine() {
    DATACARD=${1}
    echo ""
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "DATACARD=${DATACARD}"
    combine -M Significance ${DATACARD} --toys -1 --expectSignal 1
}

#
git clone https://github.com/kyungminparkdrums/CombineTool.git
run-combine CombineTool/example/VBF_WZ_card.txt

#
DATACARD_FILE=datacard.combine
python ../bin/make-datacard.py \
    -i CombineTool/example/SR/IndividualProcesses/ \
    -s ./systematics.json \
    --step NRecoBJet \
    --kinematic LargestDiJetMass \
    --rebin-edges 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4500 5000 \
    --datacard-file ${DATACARD_FILE}
run-combine  ${DATACARD_FILE}
