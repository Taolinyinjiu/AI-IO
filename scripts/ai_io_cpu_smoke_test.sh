#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_URL="https://github.com/SJTU-ViSYS-team/AI-IO/releases/download/v1.0/AI-IO_dataset.tar.gz"
CHECKPOINT_URL="https://github.com/SJTU-ViSYS-team/AI-IO/releases/download/v1.0/checkpoint_open.pt"

DATASET_ARCHIVE="${REPO_DIR}/downloads/AI-IO_dataset.tar.gz"
DATASET_EXTRACT_DIR="${REPO_DIR}/datasets"
OUT_DIR="${REPO_DIR}/results"
DATASET_NAME="our2"
CHECKPOINT_NAME="checkpoint_open.pt"
MODEL_PARAM_NAME="model_net_parameters.json"
GENERATED_CONFIG="${REPO_DIR}/config/our2_cpu_smoke.generated.conf"
DEFAULT_SEQUENCE="indoor/manual/high/seq_1"

FORCE_DOWNLOAD=0
FORCE_EXTRACT=0
ALL_SEQUENCES=0
SEQUENCE="${DEFAULT_SEQUENCE}"
SKIP_FILTER=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Download the official AI-IO dataset/checkpoint, extract them under this
repository, generate a CPU-only test config, and run src/main_filter.py --cpu.

Options:
  --force-download      Re-download dataset/checkpoint even if files exist.
  --force-extract       Remove and re-extract datasets/ before testing.
  --sequence PATH       Test one dataset sequence. Default: ${DEFAULT_SEQUENCE}
  --all-sequences       Run every sequence listed in config/our2.conf.
  --skip-filter         Download/extract/prepare only; do not run filter.
  -h, --help            Show this help.

Expected environment:
  conda activate ai-io
  python -m pip install -r requirements.txt
EOF
}

log() {
    printf '\n[AI-IO CPU test] %s\n' "$*"
}

die() {
    printf '\n[AI-IO CPU test] ERROR: %s\n' "$*" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local output="$2"
    mkdir -p "$(dirname "$output")"
    if [[ -s "$output" && "${FORCE_DOWNLOAD}" -eq 0 ]]; then
        log "Using existing file: ${output}"
        return
    fi

    log "Downloading ${url}"
    if have_cmd curl; then
        curl -L --fail --retry 3 --retry-delay 2 -o "$output" "$url"
    elif have_cmd wget; then
        wget -O "$output" "$url"
    else
        die "curl or wget is required for downloads"
    fi
}

require_python_env() {
    log "Checking Python environment"
    python - <<'PY'
import importlib.util
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "Python >= 3.10 is required for this AI-IO environment; "
        f"current is {sys.version.split()[0]}"
    )

required = ["torch", "numpy", "scipy", "numba", "h5py", "pyhocon"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nRun: python -m pip install -r requirements.txt"
    )

import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
}

extract_dataset() {
    mkdir -p "${DATASET_EXTRACT_DIR}"

    if [[ "${FORCE_EXTRACT}" -eq 1 ]]; then
        log "Removing existing extracted dataset directory: ${DATASET_EXTRACT_DIR}"
        rm -rf "${DATASET_EXTRACT_DIR}"
        mkdir -p "${DATASET_EXTRACT_DIR}"
    fi

    if find_dataset_root >/dev/null 2>&1; then
        log "Dataset already extracted under ${DATASET_EXTRACT_DIR}"
        return
    fi

    log "Extracting dataset to ${DATASET_EXTRACT_DIR}"
    tar -xzf "${DATASET_ARCHIVE}" -C "${DATASET_EXTRACT_DIR}"
}

find_dataset_root() {
    local marker
    marker="$(find "${DATASET_EXTRACT_DIR}" -path "*/${DEFAULT_SEQUENCE}/processed_data/test/data.hdf5" -print -quit 2>/dev/null || true)"
    [[ -n "${marker}" ]] || return 1
    printf '%s\n' "${marker%/${DEFAULT_SEQUENCE}/processed_data/test/data.hdf5}"
}

prepare_checkpoint() {
    local checkpoint_dir="${OUT_DIR}/${DATASET_NAME}/checkpoints/model_net"
    mkdir -p "${checkpoint_dir}"
    download_file "${CHECKPOINT_URL}" "${checkpoint_dir}/${CHECKPOINT_NAME}"

    local model_param="${checkpoint_dir}/${MODEL_PARAM_NAME}"
    if [[ ! -s "${model_param}" ]]; then
        log "Writing minimal ${MODEL_PARAM_NAME}"
        cat >"${model_param}" <<'EOF'
{
  "sampling_freq": 100,
  "window_time": 1
}
EOF
    else
        log "Using existing model parameter file: ${model_param}"
    fi
}

write_smoke_config() {
    local dataset_root="$1"

    if [[ "${ALL_SEQUENCES}" -eq 1 ]]; then
        log "Generating config for all sequences"
        python - "${REPO_DIR}/config/our2.conf" "${GENERATED_CONFIG}" "${dataset_root}" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
root = sys.argv[3]
text = src.read_text()
text = text.replace("data_root: path/to/dataset", f'data_root: "{root}"')
dst.write_text(text)
print(dst)
PY
        return
    fi

    log "Generating smoke-test config for one sequence: ${SEQUENCE}"
    cat >"${GENERATED_CONFIG}" <<EOF
test:
{
    mode: test
    data_list:
    [{
        name: "${DATASET_NAME}"
        data_root: "${dataset_root}"
        data_drive: [
            "${SEQUENCE}"
        ]
    }]
}
EOF
}

verify_sequence_exists() {
    local dataset_root="$1"
    local seq="$2"
    local data_file="${dataset_root}/${seq}/processed_data/test/data.hdf5"
    [[ -s "${data_file}" ]] || die "Expected test data not found: ${data_file}"
}

run_filter_cpu() {
    mkdir -p "${OUT_DIR}"
    log "Running AI-IO filter on CPU"
    (
        cd "${REPO_DIR}"
        export CUDA_VISIBLE_DEVICES=""
        export NUMBA_CACHE_DIR="${REPO_DIR}/.cache/numba"
        mkdir -p "${NUMBA_CACHE_DIR}"
        python src/main_filter.py \
            --data_config="${GENERATED_CONFIG}" \
            --out_dir="${OUT_DIR}" \
            --dataset="${DATASET_NAME}" \
            --checkpoint_fn="${CHECKPOINT_NAME}" \
            --model_param_fn="${MODEL_PARAM_NAME}" \
            --cpu
    )
}

print_outputs() {
    log "Output summary"
    find "${OUT_DIR}/${DATASET_NAME}" -path "*/pyfilter/stamped_traj_estimate.txt" -print | sort
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-download)
            FORCE_DOWNLOAD=1
            shift
            ;;
        --force-extract)
            FORCE_EXTRACT=1
            shift
            ;;
        --sequence)
            [[ $# -ge 2 ]] || die "--sequence requires a value"
            SEQUENCE="$2"
            shift 2
            ;;
        --all-sequences)
            ALL_SEQUENCES=1
            shift
            ;;
        --skip-filter)
            SKIP_FILTER=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ -f "${REPO_DIR}/src/main_filter.py" ]] || die "Run this script from an AI-IO checkout"

require_python_env
download_file "${DATASET_URL}" "${DATASET_ARCHIVE}"
extract_dataset
DATASET_ROOT="$(find_dataset_root)" || die "Could not locate extracted AI-IO dataset root"
log "Dataset root: ${DATASET_ROOT}"
prepare_checkpoint

if [[ "${ALL_SEQUENCES}" -eq 0 ]]; then
    verify_sequence_exists "${DATASET_ROOT}" "${SEQUENCE}"
fi

write_smoke_config "${DATASET_ROOT}"
log "Generated config: ${GENERATED_CONFIG}"

if [[ "${SKIP_FILTER}" -eq 0 ]]; then
    run_filter_cpu
    print_outputs
else
    log "Skipped filter run by request"
fi

log "Done"
