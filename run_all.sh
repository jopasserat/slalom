mkdir -p results

DATA_DIR=$1

################
## Benchmarks ##
################

python -u -m python.slalom.scripts.benchmarks > results/benchmarks.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.benchmarks --use_sgx > results/benchmarks_sgx.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.benchmarks --threads=2 > results/benchmarks_threaded_4.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.benchmarks --threads=3 > results/benchmarks_threaded_8.txt 2>&1
sleep 5

###########
## VGG16 ##
###########

# forward pass
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 --input_dir="${DATA_DIR}" > results/vgg_full_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx --input_dir="${DATA_DIR}" > results/vgg_full_sgx.txt 2>&1
sleep 5

# verify
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --input_dir="${DATA_DIR}" > results/vgg_full_verif_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --use_sgx --input_dir="${DATA_DIR}" > results/vgg_full_verif_sgx.txt 2>&1
sleep 5

# verify batched
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --verify_batched --input_dir="${DATA_DIR}" > results/vgg_full_verif_batched_cpu.txt 2>&1
sleep 5

# verify preproc
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --input_dir="${DATA_DIR}" > results/vgg_full_verif_preproc_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx --input_dir="${DATA_DIR}" > results/vgg_full_verif_preproc_sgx.txt 2>&1
sleep 5

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --input_dir="${DATA_DIR}" > results/vgg_full_slalom_privacy_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --use_sgx --input_dir="${DATA_DIR}" > results/vgg_full_slalom_privacy_sgx.txt 2>&1
sleep 5

# slalom privacy+integrity
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --integrity --input_dir="${DATA_DIR}" > results/vgg_full_slalom_privacy_integrity_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --integrity --use_sgx --input_dir="${DATA_DIR}" > results/vgg_full_slalom_privacy_integrity_sgx.txt 2>&1
sleep 5

################
## VGG No Top ##
################

# forward pass
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx --no_top --input_dir="${DATA_DIR}" > results/vgg_notop_sgx.txt 2>&1
sleep 5

# verif
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --use_sgx --no_top --verify --input_dir="${DATA_DIR}" > results/vgg_notop_verif_sgx.txt 2>&1
sleep 5

# verif preproc
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --use_sgx --no_top --verify --preproc --input_dir="${DATA_DIR}" > results/vgg_notop_verif_preproc_sgx.txt 2>&1
sleep 5

###############
## MobileNet ##
###############

# forward pass
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=1 --max_num_batches=4 --input_dir="${DATA_DIR}" > results/mobilenet_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_sgx.txt 2>&1
sleep 5

# verify
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --input_dir="${DATA_DIR}" > results/mobilenet_verif_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_verif_sgx.txt 2>&1
sleep 5

# verify batched
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --verify_batched --input_dir="${DATA_DIR}" > results/mobilenet_verif_batched_cpu.txt 2>&1
sleep 5

# verify preproc
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --input_dir="${DATA_DIR}" > results/mobilenet_verif_preproc_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_verif_preproc_sgx.txt 2>&1
sleep 5

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom mobilenet --batch_size=16 --max_num_batches=4 --blinding --input_dir="${DATA_DIR}" > results/mobilenet_slalom_privacy_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval_slalom mobilenet --batch_size=16 --max_num_batches=4 --blinding --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_slalom_privacy_sgx.txt 2>&1
sleep 5

###################
## MobileNet-Sep ##
###################

# verify preproc
python -u -m python.slalom.scripts.eval mobilenet_sep sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --input_dir="${DATA_DIR}" > results/mobilenet_sep_verif_preproc_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval mobilenet_sep sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_sep_verif_preproc_sgx.txt 2>&1
sleep 5

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --input_dir="${DATA_DIR}" > results/mobilenet_sep_slalom_privacy_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_sep_slalom_privacy_sgx.txt 2>&1
sleep 5

# slalom privacy+integrity
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --integrity --input_dir="${DATA_DIR}" > results/mobilenet_sep_slalom_privacy_integrity_cpu.txt 2>&1
sleep 5
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --integrity --use_sgx --input_dir="${DATA_DIR}" > results/mobilenet_sep_slalom_privacy_integrity_sgx.txt 2>&1
sleep 5

