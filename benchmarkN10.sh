for net in HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN; do
    python scripts/benchmark/PDIF.py --key $net --cfg-file benchmark10_causal.yml
done

for net in HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN; do
    python scripts/benchmark/DDC.py --key $net --cfg-file benchmark10_causal.yml
done

for net in HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN; do
    python scripts/benchmark/STE.py --key $net --cfg-file benchmark10_causal.yml
done

for net in HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN; do
    python scripts/benchmark/GLMCC.py --key $net --cfg-file benchmark10_causal.yml
done

for ccm in CCM FDCCM SCCM; do
    for net in HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN; do
        python scripts/benchmark/CCM.py --key $net --ccm $ccm --cfg-file benchmark10_causal.yml
    done
done